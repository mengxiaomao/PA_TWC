# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 14:38:15 2018

@author: lenovo
"""
import numpy as np
from Environment_CU import Env_cellular
from Benchmark_alg import Benchmark_alg

#fd = 10
Ts = 20e-3
n_x = 5
n_y = 5
L = 2
C = 16
#maxM = 4   # user number in one BS
min_dis = 0.01 #km
#max_dis = 1. #km
max_p = 38. #dBm
p_n = -114. #dBm
power_num = 1

    
def Benchmark_test(max_episode, Ns, fd, max_dis, maxM):
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    bench = Benchmark_alg(env)
    env.set_Ns(Ns)
    reward_hist = list()
    for k in range(1, max_episode+1):  
        reward_list = list()
        s = env.reset__()
        for i in range(int(Ns)-1):
            p = bench.calculate(s)
            s_next, r = env.step__(p)
            s = s_next
            reward_list.append(r)
        reward_hist.append(reward_list)
        
    reward_hist = np.reshape(reward_hist, [max_episode, Ns-1, 4])
    reward_hist = np.nanmean(np.nanmean(reward_hist, 0), 0)
    return reward_hist
    
def Benchmark_test_time(max_episode, Ns, fd, max_dis, maxM):
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    bench = Benchmark_alg(env)
    env.set_Ns(Ns)
    time_cost_1 = 0
    time_cost_2 = 0
    for k in range(1, max_episode+1):
        s = env.reset__()
        for i in range(int(Ns)-1):
            tc, p = bench.time_cost(s)
            time_cost_1 = time_cost_1 + tc[0]
            time_cost_2 = time_cost_2 + tc[1]
            s_next, _ = env.step__(p)
            s = s_next
        
    return [time_cost_1, time_cost_2]