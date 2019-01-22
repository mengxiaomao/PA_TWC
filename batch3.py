# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:04:44 2018
cell range R/ doppler f_d/ AP density K
feature 2
@author: mengxiaomao / Email: mengxiaomaomao@outlook.com
"""
from policy02 import Test_policy_all, Test_policy_time
from dqn_net06 import Test_dqn_all, Test_dqn_time
from ddpg02 import Test_ddpg_all, Test_ddpg_time
from Benchmark_test import Benchmark_test, Benchmark_test_time
import numpy as np
import scipy.io as sc

num = 3
name1 = './PA_RL/policy_'
name2 = './PA_RL/dqn_'
name3 = './PA_RL/ddpg_'

name_list1 = list()
name_list2 = list()
name_list3 = list()
for i in range(num):
    name_list1.append(name1 + str(i)+'.mat')
    name_list2.append(name2 + str(i)+'.mat')
    name_list3.append(name3 + str(i)+'.mat') 
        
max_episode = 500
Ns = 300+1



## max_dis 
#fd = 10
#maxM = 4
#max_dis = 1.
#
#max_dis = [0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.2,1.5]
#value1 = np.zeros((num, len(max_dis)), dtype = np.float32)
#value2 = np.zeros((num, len(max_dis)), dtype = np.float32)
#value3 = np.zeros((num, len(max_dis)), dtype = np.float32)
#for i in range(len(max_dis)):
#    for k in range(num):    
#        print(k,i)
#        weight_file1 = name_list1[k]
#        value1[k,i] = Test_policy_all(weight_file1, max_episode, Ns, fd, max_dis[i], maxM)       
#        weight_file2 = name_list2[k]
#        value2[k,i] = Test_dqn_all(weight_file2, max_episode, Ns, fd, max_dis[i], maxM)   
#        weight_file3 = name_list3[k]
#        value3[k,i] = Test_ddpg_all(weight_file3, max_episode, Ns, fd, max_dis[i], maxM)
#
#file_name='./results/test_max_dis.mat'  
#sc.savemat(file_name, {'value1': value1, 'value2': value2, 'value3': value3})  


## maxM
#fd = 10
#maxM = 4
#max_dis = 1.
#
#maxM = [1,2,3,5,6,7,8]
#value1 = np.zeros((num, len(maxM)), dtype = np.float32)
#value2 = np.zeros((num, len(maxM)), dtype = np.float32)
#value3 = np.zeros((num, len(maxM)), dtype = np.float32)
#for i in range(len(maxM)):
#    for k in range(num):    
#        print(k,i)
#        weight_file1 = name_list1[k]
#        value1[k,i] = Test_policy_all(weight_file1, max_episode, Ns, fd, max_dis, maxM[i])       
#        weight_file2 = name_list2[k]
#        value2[k,i] = Test_dqn_all(weight_file2, max_episode, Ns, fd, max_dis, maxM[i])   
#        weight_file3 = name_list3[k]
#        value3[k,i] = Test_ddpg_all(weight_file3, max_episode, Ns, fd, max_dis, maxM[i])
#
#file_name='./results/test_maxM.mat'  
#sc.savemat(file_name, {'value1': value1, 'value2': value2, 'value3': value3})

## fd
#fd = 10
#maxM = 4
#max_dis = 1.
#
#fd = [4,6,8,12,14,16,18]
#value1 = np.zeros((num, len(fd)), dtype = np.float32)
#value2 = np.zeros((num, len(fd)), dtype = np.float32)
#value3 = np.zeros((num, len(fd)), dtype = np.float32)
#for i in range(len(fd)):
#    for k in range(num):    
#        print(k,i)
#        weight_file1 = name_list1[k]
#        value1[k,i] = Test_policy_all(weight_file1, max_episode, Ns, fd[i], max_dis, maxM)       
#        weight_file2 = name_list2[k]
#        value2[k,i] = Test_dqn_all(weight_file2, max_episode, Ns, fd[i], max_dis, maxM)   
#        weight_file3 = name_list3[k]
#        value3[k,i] = Test_ddpg_all(weight_file3, max_episode, Ns, fd[i], max_dis, maxM)
#
#file_name='./results/test_fd.mat'  
#sc.savemat(file_name, {'value1': value1, 'value2': value2, 'value3': value3})



#num = 4
#max_episode = 500
#Ns = 300+1
#
## max_dis 
#fd = 10
#maxM = 4
#max_dis = 1.
#max_dis = [0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.2,1.5]
#value = np.zeros((num, len(max_dis)), dtype = np.float32)
#for i in range(len(max_dis)):   
#    value[:,i] = Benchmark_test(max_episode, Ns, fd, max_dis[i], maxM)
#file_name='./results/test_max_dis_bench.mat'  
#sc.savemat(file_name, {'value': value})  
#
## maxM
#fd = 10
#maxM = 4
#max_dis = 1.
#maxM = [1,2,3,5,6,7,8]
#value = np.zeros((num, len(maxM)), dtype = np.float32)
#for i in range(len(maxM)):
#    value[:,i] = Benchmark_test(max_episode, Ns, fd, max_dis, maxM[i])
#file_name='./results/test_maxM_bench.mat'  
#sc.savemat(file_name, {'value': value})  

num = 3

max_episode = 500
Ns = 200+1

fd = 10
maxM = 4
max_dis = 1.

value1 = np.zeros((num), dtype = np.float32)
value2 = np.zeros((num), dtype = np.float32)
value3 = np.zeros((num), dtype = np.float32)
for k in range(num):
    weight_file1 = name_list1[k]
    value1[k] = Test_policy_time(weight_file1, max_episode, Ns, fd, max_dis, maxM)       
    weight_file2 = name_list2[k]
    value2[k] = Test_dqn_time(weight_file2, max_episode, Ns, fd, max_dis, maxM)   
    weight_file3 = name_list3[k]
    value3[k] = Test_ddpg_time(weight_file3, max_episode, Ns, fd, max_dis, maxM)
    
value4 = Benchmark_test_time(max_episode, Ns, fd, max_dis, maxM)

file_name='./results/test_time_cost.mat'  
sc.savemat(file_name, {'value1': value1, 'value2': value2, 'value3': value3, 'value4': value4}) 