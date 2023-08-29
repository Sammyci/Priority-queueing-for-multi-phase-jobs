import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
def cdf(y,alpha,beta):
    gamma = (beta-1) / (alpha ** (1-beta))
    return ((alpha ** (1-beta)) + ((1-beta) * y / gamma)) ** (1 / (1 - beta))

def generate_jobs(timeend, lamb, a2l, a2u, visitProb, beta, alpha, random_seed = 42):#reproducebole
    np.random.seed(random_seed)

    a1k = np.random.exponential(1/lamb)
    a2k = np.random.uniform(a2l, a2u)
    arrivalTimes = [a1k * a2k]
    #Accumulate to get arrival time
    while True:
        next_arrival = arrivalTimes[-1] + (np.random.uniform(a2l,a2k) * np.random.exponential(1/lamb))
        if next_arrival > timeend:
            break
        arrivalTimes.append(next_arrival)
    jobNum = len(arrivalTimes)
    maxSubjobs = len(visitProb)
    subjobs = np.random.choice(range(1, maxSubjobs+1), size = jobNum , p = visitProb)
    # generate subjobs servicetime
    jobs = {}
    for i in range(jobNum):
        Y = [np.random.uniform() for _ in range(subjobs[i])]
        service = [cdf(y,alpha,beta) for y in Y]
        jobs[i+1] = [arrivalTimes[i], service]
    return jobs


def trace(interArrival, serviceTime, n, h):
    # Accumulated service time is given in the question so it has to be added up
    arrivalTime = np.cumsum(interArrival)
    arrivalTime = [round(x, 4) for x in arrivalTime]
    arrivalIndex = {arrivalTime[i]: i+1 for i in range(len(arrivalTime))}
    jobService = {(i+1):[arrivalTime[i],serviceTime[i]] for i in range(len(serviceTime))}
    jobStatus = {(i+1):[0,len(serviceTime[i])] for i in range(len(serviceTime))}

    response_cumulative=0
    response_info= {}
    departure_info = []

    next_arrival_time = arrivalTime[0]
    next_departure_time = math.inf

    highPriority = []
    lowPriority = []

    master_clock = 0
    #All servers are idle at the start
    servers_busy = [0] * n



    while True:
        if next_arrival_time < next_departure_time:
            next_job_time = next_arrival_time
            next_job_type = "arrival"
        else:
            next_job_time = next_departure_time
            next_job_type = "departure"

        master_clock = next_job_time
        #If the work is an external task
        if next_job_type == "arrival":
            jobIndex = arrivalIndex[next_job_time]

            queue_element = [(jobIndex,jobStatus[jobIndex]), next_arrival_time, jobService[jobIndex][1]]
            highPriority.append(queue_element)

            if 0 in servers_busy:
                idle_index = servers_busy.index(0)

                if highPriority or lowPriority:
                    if highPriority:
                        eventStatus = highPriority.pop(0)
                    elif not highPriority and lowPriority:
                        eventStatus = lowPriority.pop(0)
                    job_completed_time = master_clock + eventStatus[-1][0]
                    eventStatus[-1].pop(0)
                    serverstate = [job_completed_time, eventStatus[1],eventStatus[-1],eventStatus[0]]
                    servers_busy[idle_index] = serverstate



            lastest_depart_time = math.inf
            for i in range(len(servers_busy)):
                if servers_busy[i] !=0:
                    if servers_busy[i][0] < lastest_depart_time:
                        lastest_depart_time = servers_busy[i][0]
                        departure_status = servers_busy[i]
                        latest_busy_server = i
            arrivalTime = np.delete(arrivalTime,0)
            if len(arrivalTime)>0:
                next_arrival_time = arrivalTime[0]
            else:
                next_arrival_time = math.inf
            next_departure_time = lastest_depart_time

        # If a job go back to depature from a server
        elif next_job_type == "departure":
            c_time , a_time , rest_service_time , j_status = departure_status
            j_idx = j_status[0]
            jobStatus[j_idx][0] += 1
            departure_info.append((a_time, c_time, jobStatus[j_idx][0], jobStatus[j_idx][1]))
            #clear servers status
            servers_busy[latest_busy_server] = 0
            if  jobStatus[j_idx][0] < jobStatus[j_idx][1]:
                queue_element = [(j_idx, jobStatus[j_idx]), a_time, rest_service_time]
                if jobStatus[j_idx][0] >= h:
                    lowPriority.append(queue_element)
                else:
                    highPriority.append(queue_element)
            else:
                #Direct calculate time
                response_cumulative += c_time - a_time


            if 0 in servers_busy:
                idle_index = servers_busy.index(0)
                if highPriority or lowPriority:
                    if highPriority:
                        eventStatus = highPriority.pop(0)
                    elif not highPriority and lowPriority:
                        eventStatus = lowPriority.pop(0)
                    job_completed_time = master_clock + eventStatus[-1][0]
                    eventStatus[-1].pop(0)
                    serverstate = [job_completed_time, eventStatus[1], eventStatus[-1], eventStatus[0]]
                    servers_busy[idle_index] = serverstate

            next_arrival_time = next_arrival_time

            lastest_depart_time = math.inf
            for i in range(len(servers_busy)):
                if servers_busy[i] != 0:
                    if servers_busy[i][0] < lastest_depart_time:
                        lastest_depart_time = servers_busy[i][0]
                        departure_status = servers_busy[i]
                        latest_busy_server = i

            next_departure_time = lastest_depart_time
        indictor = True
        for k,v in jobStatus.items():
            if v[0] != v[1]:
                indictor = False
        if indictor:
            break
    return response_cumulative/ len(serviceTime), departure_info

def random(jobs, n, h, timeEnd):
    jobStatus = {(i+1):[0,len(jobs[i+1][1])] for i in range(len(jobs))}
    arrivalIndex = {jobs[i+1][0]: i+1 for i in range(len(jobs))}
    response_cumulative = 0
    highPriority = []
    lowPriority = []

    master_clock = 0
    servers_busy = [0] * n

    departure_status = 0
    latest_busy_server = 0

    next_arrival_time = min(jobs[i+1][0] for i in range(len(jobs)))
    next_departure_time = math.inf

    departure_info = []

    while master_clock < timeEnd:
        if next_arrival_time < next_departure_time:
            next_job_time = next_arrival_time
            next_job_type = "arrival"
        else:
            next_job_time = next_departure_time
            next_job_type = "departure"

        master_clock = next_job_time

        if next_job_type == "arrival":
            jobIndex = arrivalIndex[next_job_time]
            queue_element = [(jobIndex, jobStatus[jobIndex]), next_arrival_time, jobs[jobIndex][1]]
            highPriority.append(queue_element)

            if 0 in servers_busy:
                idle_index = servers_busy.index(0)
                if highPriority or lowPriority:
                    if highPriority:

                        eventStatus = highPriority.pop(0)
                    elif not highPriority and lowPriority:

                        eventStatus = lowPriority.pop(0)
                    job_completed_time = master_clock + eventStatus[-1][0]
                    eventStatus[-1].pop(0)
                    serverstate = [job_completed_time, eventStatus[1], eventStatus[-1], eventStatus[0]]
                    servers_busy[idle_index] = serverstate

            lastest_depart_time = math.inf
            for i in range(len(servers_busy)):
                if servers_busy[i] != 0:
                    if servers_busy[i][0] < lastest_depart_time:
                        lastest_depart_time = servers_busy[i][0]
                        departure_status = servers_busy[i]
                        latest_busy_server = i

            next_arrival_time = min([jobs[i + 1][0] for i in range(len(jobs)) if jobs[i + 1][0] > master_clock],
                                    default=math.inf)
            next_departure_time = lastest_depart_time


        elif next_job_type == "departure":
            c_time, a_time, rest_service_time, j_status = departure_status
            j_idx = j_status[0]
            jobStatus[j_idx][0] += 1
            departure_info.append((a_time, c_time, jobStatus[j_idx][0], jobStatus[j_idx][1]))

            if jobStatus[j_idx][0] < jobStatus[j_idx][1]:
                queue_element = [(j_idx, jobStatus[j_idx]), a_time, rest_service_time]
                if jobStatus[j_idx][0] >= h:
                    lowPriority.append(queue_element)
                else:
                    highPriority.append(queue_element)
            else:
                response_cumulative += c_time - a_time

            servers_busy[latest_busy_server] = 0

            if 0 in servers_busy:
                idle_index = servers_busy.index(0)
                if highPriority or lowPriority:
                    if highPriority:
                        eventStatus = highPriority.pop(0)
                    elif not highPriority and lowPriority:
                        eventStatus = lowPriority.pop(0)
                    job_completed_time = master_clock + eventStatus[-1][0]
                    eventStatus[-1].pop(0)
                    serverstate = [job_completed_time, eventStatus[1], eventStatus[-1], eventStatus[0]]
                    servers_busy[idle_index] = serverstate

                lastest_depart_time = math.inf
                for i in range(len(servers_busy)):
                    if servers_busy[i] != 0:
                        if servers_busy[i][0] < lastest_depart_time:
                            lastest_depart_time = servers_busy[i][0]
                            departure_status = servers_busy[i]
                            latest_busy_server = i

                next_arrival_time = min([jobs[i + 1][0] for i in range(len(jobs)) if jobs[i + 1][0] > master_clock],
                                        default=math.inf)
                next_departure_time = lastest_depart_time

    remaining_jobs = sum([jobStatus[job][1] - jobStatus[job][0] for job in jobStatus])
    if remaining_jobs != 0:
        response_cumulative /= (len(jobs) - remaining_jobs)
    else:
        response_cumulative /= len(jobs)

    return response_cumulative, departure_info

max = 5
n_values = list(range(1, 6))
mrt_results = {}
departresults = {}
mean_response_time = {}
std_dev_response_time = {}
confidence_intervals = {}
n = 2
for h in range(1,max):
    for i in range(5):
        jobs = generate_jobs(1200, 1.8, 0.6, 1.1, [0.4, 0.3, 0.2, 0.07, 0.03], 4.4, 0.2)
        mrt, dep = random(jobs, n, h, 1200)
        dep_list = [tup[1] - tup[0] for tup in dep]
        mrt_results[h] = mrt
        departresults[h] = dep_list
    mean_response_time[h] = np.mean(departresults[h][20:])
    std_dev_response_time[h] = np.std(departresults[h][20:])
alpha = 0.05
degrees_freedom = len(departresults[h][20:]) - 1
t_score = stats.t.ppf(1 - alpha / 2, degrees_freedom)

for h in range(1, max):
    margin_of_error = t_score * (std_dev_response_time[h] / np.sqrt(len(departresults[h][20:])))
    confidence_intervals[h] = (mean_response_time[h] - margin_of_error, mean_response_time[h] + margin_of_error)

best_h = min(confidence_intervals, key=lambda x: confidence_intervals[x][1])
best_mrt = mrt_results[best_h]
h_values = list(range(1, max))
lower_bounds = [confidence_intervals[h][0] for h in h_values]
upper_bounds = [confidence_intervals[h][1] for h in h_values]
means = [mean_response_time[h] for h in h_values]

plt.figure(figsize=(10, 6))
plt.plot(h_values, means, marker='o', label='Average response time')
plt.fill_between(h_values, lower_bounds, upper_bounds, alpha=0.2, label='95% confidence interval')
plt.axhline(best_mrt, color='r', linestyle='--', label=f'Best average response time：{best_mrt:.2f}')
plt.axvline(best_h, color='g', linestyle='--', label=f'Optimal threshold h：{best_h}')

plt.xlabel('Threshold h')
plt.ylabel('Average response time')
plt.legend()
plt.title('Mean response times for different h values and their 95% confidence intervals')
plt.grid()
plt.show()
print("Optimal threshold h：", best_h)