import numpy as np
import math
import sys
import os
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




def main(s):
    mode = [row.strip() for row in open(f"config/mode_{s}.txt").readlines()][0]

    parasFiles = [row.strip() for row in open(f"config/para_{s}.txt").readlines()]
    arrivalFiles= [row.strip() for row in open(f"config/interarrival_{s}.txt").readlines()]
    serviceFiles = np.loadtxt(f"config/service_{s}.txt")
    out_folder = 'output'

    if mode == "trace":
        n = int(parasFiles[0])
        h = int(parasFiles[1])
        interArrival = list(float(a) for a in arrivalFiles)
        serviceTime = list(list(t for t in job if not np.isnan(t)) for job in serviceFiles)
        mrt,dep = trace(interArrival, serviceTime, n, h)
        mrt_file = os.path.join(out_folder, f'mrt_{s}.txt')
        dep_file = os.path.join(out_folder, f'dep_{s}.txt')

    elif mode == "random":
        n,h,timeEnd = int (parasFiles[0]), int(parasFiles[1]), int(parasFiles[2])
        lamb, alpha_2l, alpha_2u = tuple(float(e) for e in arrivalFiles[0].split())
        visitProb = list (float(t) for t in arrivalFiles[1].split())
        beta, alpha = tuple(serviceFiles)
        jobs = generate_jobs(timeEnd, lamb, alpha_2l, alpha_2u, visitProb, beta, alpha, random_seed = 42)
        mrt, dep = random(jobs, n, h, timeEnd)
        mrt_file = os.path.join(out_folder, f'mrt_{s}.txt')
        dep_file = os.path.join(out_folder, f'dep_{s}.txt')


    # Save mean response time
    with open(mrt_file, 'w') as file:
        file.write(f'{mrt:.4f}\n')

    # Save completion times
    with open(dep_file, 'w') as file:
        for line in dep:
            file.write(f'{line[0]:.4f} {line[1]:.4f} {line[2]} {line[3]}\n')

if __name__ == "__main__":
   main(sys.argv[1])