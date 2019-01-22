# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 11:24:43 2018
Q / gamma = 0
Baseline 2 feature:
Baseline 3 feature: 1.535 1.514 1.520 1.505
Using Experience_Replay
@author: mengxiaomao / Email: mengxiaomaomao@outlook.com
"""
import time
import numpy as np
import tensorflow as tf
from DQN import DNN, DQN
from Environment_CU import Env_cellular
from Experience_replay import ReplayBuffer

fd = 10
Ts = 20e-3
n_x = 5
n_y = 5
L = 2
C = 16
maxM = 4   # user number in one BS
min_dis = 0.01 #km
max_dis = 1. #km
max_p = 38. #dBm
p_n = -114. #dBm
power_num = 10  #action_num

def Train(sess, env, weight_file):
    max_reward = 0
    batch_size = 500
    max_episode = 5000
    buffer_size = 50000
    Ns = 11
    env.set_Ns(Ns)  
    dnn = DNN(env, weight_file, max_episode = max_episode)
    dqn = DQN(sess, dnn)
    
    tf.global_variables_initializer().run()
    deque = ReplayBuffer(buffer_size)
    interval = 100
    st = time.time()
    reward_hist = list()
    for k in range(1, max_episode+1):  
        reward_dqn_list = list()
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            a = dqn.predict_a(s_actor)
            p, a = dqn.select_action(a, k)
            s_actor_next, _, rate, r = env.step(p)
            deque.add(s_actor, a, rate)
            s_actor = s_actor_next
            reward_dqn_list.append(r)
        if deque.size() > batch_size:
            batch_s, batch_a, batch_r = deque.sample_batch(batch_size)
            dqn.train(batch_s, batch_a, batch_r)
            
        reward_hist.append(np.mean(reward_dqn_list))   # bps/Hz per link
        if k % interval == 0: 
            reward = np.mean(reward_hist[-interval:])
            if reward > max_reward:
                dnn.save_params()
                max_reward = reward
            print("Episode(train):%d  DQN: %.3f  Time cost: %.2fs" 
                  %(k, reward, time.time()-st))
            st = time.time()
    return reward_hist
    
def Test(sess, env, weight_file):
    max_episode = 100
    Ns = 5e2+1
    env.set_Ns(Ns) 
    dnn = DNN(env, weight_file)
    dqn = DQN(sess, dnn)
    
    tf.global_variables_initializer().run()
    dqn.load_params()
    st = time.time()
    reward_hist = list()
    for k in range(1, max_episode+1):  
        reward_dqn_list = list()
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            p = dqn.predict_p(s_actor)
            s_actor_next, _, _, r = env.step(p)
            s_actor = s_actor_next
            reward_dqn_list.append(r)
        reward_hist.append(np.mean(reward_dqn_list))   # bps/Hz per link
        print("Episode(test):%d  DQN: %.3f  Time cost: %.2fs" 
              %(k, reward_hist[-1], time.time()-st))
        st = time.time()
    print("Test average rate: %.3f" %(np.mean(reward_hist)))
    return reward_hist
    
def Test_dqn_mem(weight_file, max_episode, Ns, fd, max_dis, maxM):
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    tf.reset_default_graph()
    with tf.Session() as sess:
        env.set_Ns(Ns) 
        dnn = DNN(env, weight_file)
        dqn = DQN(sess, dnn)
        
        tf.global_variables_initializer().run()
        dqn.load_params()
        reward_hist = list()
        for k in range(1, max_episode+1):  
            reward_dqn_list = list()
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                p = dqn.predict_p(s_actor)
                s_actor_next, _, _, r = env.step(p)
                s_actor = s_actor_next
                reward_dqn_list.append(r)
            reward_hist.append(np.mean(reward_dqn_list))   # bps/Hz per link
    return np.mean(reward_hist)
    
def Test_dqn_all(weight_file, max_episode, Ns, fd, max_dis, maxM):
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    tf.reset_default_graph()
    with tf.Session() as sess:
        env.set_Ns(Ns) 
        dnn = DNN(env, weight_file)
        dqn = DQN(sess, dnn)
        
        tf.global_variables_initializer().run()
        dqn.load_params()
        reward_hist = list()
        for k in range(1, max_episode+1):  
            reward_dqn_list = list()
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                p = dqn.predict_p(s_actor)
                s_actor_next, _, _, r = env.step(p)
                s_actor = s_actor_next
                reward_dqn_list.append(r)
            reward_hist.append(np.mean(reward_dqn_list))   # bps/Hz per link
    return np.mean(reward_hist)

def Test_dqn_time(weight_file, max_episode, Ns, fd, max_dis, maxM):
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    tf.reset_default_graph()
    with tf.Session() as sess:
        env.set_Ns(Ns) 
        dnn = DNN(env, weight_file)
        dqn = DQN(sess, dnn)
        
        tf.global_variables_initializer().run()
        dqn.load_params()
        time_cost = 0
        for k in range(1, max_episode+1):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                st = time.time()
                p = dqn.predict_p(s_actor)
                time_cost = time_cost + time.time() - st
                s_actor_next, _, _, _ = env.step(p)
                s_actor = s_actor_next
    return time_cost
    
def Test_dqn_mem_quan(weight_file, max_episode, Ns, power_num, fd, max_dis, maxM):
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    tf.reset_default_graph()
    with tf.Session() as sess:
        env.set_Ns(Ns) 
        dnn = DNN(env, weight_file)
        dqn = DQN(sess, dnn)
        
        tf.global_variables_initializer().run()
        dqn.load_params()
        reward_hist = list()
        for k in range(1, max_episode+1):  
            reward_dqn_list = list()
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                p = dqn.predict_p(s_actor)
                s_actor_next, _, _, r = env.step(p)
                s_actor = s_actor_next
                reward_dqn_list.append(r)
            reward_hist.append(np.mean(reward_dqn_list))   # bps/Hz per link
    return np.mean(reward_hist)
    
def Train_dqn_mem(weight_file, fd, max_dis, maxM):
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    tf.reset_default_graph()
    with tf.Session() as sess:
        Train(sess, env, weight_file)
        
def Train_dqn_mem_quan(weight_file, power_num, fd, max_dis, maxM):
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    tf.reset_default_graph()
    with tf.Session() as sess:
        Train(sess, env, weight_file)
        
if __name__ == "__main__":
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    weight_file = 'C:/Software/workshop/python/dqn_6.mat'
    tf.reset_default_graph()
    with tf.Session() as sess:
        train_hist = Train(sess, env, weight_file)
        
    tf.reset_default_graph()
    with tf.Session() as sess:
        test_hist = Test(sess, env, weight_file)
            
            
            
            