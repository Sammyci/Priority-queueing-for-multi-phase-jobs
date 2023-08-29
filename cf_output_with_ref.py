#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMP9334 Project sample file 

This file compares the output files against their reference.

For trace mode, it checks mrt_*.txt and dep_*.txt

For random mode, it checks mrt_*.txt only 

Assumptions on file location: This file assumes that the output/ and ref/
sub-directories are below the directory that this file is located.
    
    Directory containing cf_output_with_ref.py
    |
    |
    |---- Sub-directory: config
    |---- Sub-directory: output
    |---- Sub-directory: ref
    

This version: 25 March 2023

@author: ctchou
"""

# import sys for input argument 
import sys
import os 

# import numpy for easy comparison 
import numpy as np

def main():
    
    # Check whether there is an input argument 
    if len(sys.argv) == 2:
        t = int(sys.argv[1])
    else:
        print('Error: Expect the test number as the input argument')
        print('Example usage: python3 cf_output_with_ref.py 1')   
        return
    
    # Location of the folders
    out_folder = 'output'
    ref_folder = 'ref'
        
    # Definitions
    file_ext = '.txt' # File extension
    
    # For trace mode, an absolute tolerance is used
    ABS_TOL = 1e-3  # Absolute tolerance 
    
    # For tests 7, 8, 9 and 10 (which are in radnom mode), 
    # the mean response time is expected to be within the range     
    MRT_TOL = { 
                 7: [0.8834, 1.0541], 
                 8: [0.7104, 0.8530],
                 9: [0.5814, 0.6586],    
                10: [0.7168, 0.7551]       
                }
    
    
    # Read test number from the input argument 
       
    # t is the test number
    # Tests 1 to 6 are trace mode
    # Tests 7 to 10 are is radnom mode
    trace_mode_test_start = 1
    trace_mode_test_end = 6 # Tests 1 to 6 
    random_mode_test_start = 7
    random_mode_test_end = 10    
    
    if t in range(trace_mode_test_start,trace_mode_test_end+1): 
    
        # Compare mrt against the reference
        out_file = os.path.join(out_folder,'mrt_'+str(t)+file_ext)
        ref_file = os.path.join(ref_folder,'mrt_'+str(t)+'_ref'+file_ext)
        
        if os.path.isfile(out_file):
            mrt_stu = np.loadtxt(out_file)
        else:
            print('Error: File ',out_file,'does NOT exist')    
            return
        
        if os.path.isfile(ref_file): 
            mrt_ref = np.loadtxt(ref_file)
        else:
            print('Error: File ',ref_file,'does NOT exist')    
            return           
        
        if np.isclose(mrt_stu,mrt_ref,atol=ABS_TOL):
            print('Test '+str(t)+': Mean response time matches the reference')
        else: 
            print('Test '+str(t)+': Mean response time does NOT match the reference')
    
        # Compare dep against the reference  
        out_file = os.path.join(out_folder,'dep_'+str(t)+file_ext)
        ref_file = os.path.join(ref_folder,'dep_'+str(t)+'_ref'+file_ext)
        
        if os.path.isfile(out_file):
            dep_stu = np.loadtxt(out_file)
        else:
            print('Error: File ',out_file,'does NOT exist')    
            return            
        
        if os.path.isfile(ref_file):
            dep_ref = np.loadtxt(ref_file)
        else:
            print('Error: File ',ref_file,'does NOT exist')    
            return                      
        
        if np.all(np.isclose(dep_stu,dep_ref,atol=ABS_TOL)):
            print('Test '+str(t)+': Completion times match the reference')
        else: 
            print('Test '+str(t)+': Completion times do NOT match the reference')
    
  
    
    elif t in range(random_mode_test_start,random_mode_test_end+1): 
        out_file = os.path.join(out_folder,'mrt_'+str(t)+file_ext)
       
        if os.path.isfile(out_file):
            mrt_stu = np.loadtxt(out_file)
        else:
            print('Error: File ',out_file,'does NOT exist')    
            return        
    
        
        if MRT_TOL[t][0] <= mrt_stu <= MRT_TOL[t][1]:
            print('Test '+str(t)+': Mean response time is within tolerance')
        else: 
            print('Test '+str(t)+': Mean response time is NOT within tolerance')
            print('You should try to run a new simulation round with new random numbers.')
            print('Your output need to be within the tolerance for most of the rounds.')
        

    else:
        print('The input argument is not a valid test number')
        
if __name__ == '__main__':
    dep_error = main()            