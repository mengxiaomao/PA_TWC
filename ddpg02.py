# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 10:33:19 2018
DDGP CU action-critic / gradient
self.cost = - self.q
self.cost = tf.nn.l2_loss(self.q_tar - self.q)
Baseline 2 feature: 1.525
Baseline 3 feature: 1.751
@author: mengxiaomao
"""
import time
import numpy as np
import tensorflow as tf
from Environment_CU import Env_cellular
from DDPG_2 import DDPG, Actor, Critic

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
power_num = 1
 
def Train(sess, env, weight_file):
    max_reward = 0
    max_episode = 5000
    Ns = 11
    env.set_Ns(Ns)
    dnn = DDPG(env, weight_file)
    actor = Actor(sess, dnn)
    critic = Critic(sess, dnn)

    tf.global_variables_initializer().run()
    interval = 100
    st = time.time()
    reward_hist = list()
    loss_hist = list()
    for k in range(1, max_episode+1):  
        reward_list = list()
        loss_list = list()
        s_actor, s_critic = env.reset()
        for i in range(int(Ns)-1):
            p, p_exp = actor.get_random_action(s_actor, k)
            s_actor_next, s_critic_next, q_tar, r = env.step(p_exp[:,0])
            loss = critic.train(s_critic, p_exp, q_tar) 
            grads = critic.get_gradient(s_critic, p)
            actor.train(s_actor, grads[0])
            s_actor, s_critic = s_actor_next, s_critic_next
            reward_list.append(r)
            loss_list.append(loss)
        reward_hist.append(np.mean(reward_list))
        loss_hist.append(np.mean(loss_list))
        if k % interval == 0: 
            reward = np.mean(reward_hist[-interval:])
            if reward > max_reward:
                dnn.save_params()
                max_reward = reward
            print("Episode(train):%d   DDPG: %.3f  Loss: %.3f  Time cost: %.2fs" 
                  %(k, reward, np.mean(loss_hist[-interval:]), time.time()-st))
            st = time.time() 
    return reward_hist

def Test(sess, env, weight_file):
    max_episode = 100
    Ns = 5e2+1
    env.set_Ns(Ns)
    dnn = DDPG(env, weight_file)
    actor = Actor(sess, dnn)
    critic = Critic(sess, dnn)
    
    tf.global_variables_initializer().run()
    actor.load_params()
    critic.load_params()
    st = time.time()
    reward_hist = list()
    for k in range(1, max_episode+1):  
        reward_list = list()
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            p = actor.predict_p(s_actor)
            s_actor_next, _, _, r = env.step(p[:,0])
            s_actor = s_actor_next  
            reward_list.append(r)
        reward_hist.append(np.mean(reward_list))
        print("Episode(test):%d  DDPG: %.3f  Time cost: %.2fs" 
              %(k, reward_hist[-1], time.time()-st))
        st = time.time()
    print("Test average rate: %.3f" %(np.mean(reward_hist)))
    return reward_hist
    
def Train_ddpg(weight_file, fd, max_dis, maxM):
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    tf.reset_default_graph()
    with tf.Session() as sess:
        Train(sess, env, weight_file)
    
def Test_ddpg(weight_file, max_episode, Ns, fd, max_dis, maxM):
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    tf.reset_default_graph()
    with tf.Session() as sess:
        env.set_Ns(Ns) 
        dnn = DDPG(env, weight_file)
        actor = Actor(sess, dnn)
        
        tf.global_variables_initializer().run()
        actor.load_params()
        reward_hist = list()
        for k in range(1, max_episode+1):  
            reward_list = list()
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                p = actor.predict_p(s_actor)
                s_actor_next, _, _, r = env.step(p[:,0])
                s_actor = s_actor_next  
                reward_list.append(r)
            reward_hist.append(np.mean(reward_list))
    return np.mean(reward_hist) 
    
def Test_ddpg_all(weight_file, max_episode, Ns, fd, max_dis, maxM):
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    tf.reset_default_graph()
    with tf.Session() as sess:
        env.set_Ns(Ns) 
        dnn = DDPG(env, weight_file)
        actor = Actor(sess, dnn)
        
        tf.global_variables_initializer().run()
        actor.load_params()
        reward_hist = list()
        for k in range(1, max_episode+1):  
            reward_list = list()
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                p = actor.predict_p(s_actor)
                s_actor_next, _, _, r = env.step(p[:,0])
                s_actor = s_actor_next  
                reward_list.append(r)
            reward_hist.append(np.mean(reward_list))
    return np.mean(reward_hist)
    
def Test_ddpg_time(weight_file, max_episode, Ns, fd, max_dis, maxM):
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    tf.reset_default_graph()
    with tf.Session() as sess:
        env.set_Ns(Ns) 
        dnn = DDPG(env, weight_file)
        actor = Actor(sess, dnn)
        
        tf.global_variables_initializer().run()
        actor.load_params()
        time_cost = 0
        for k in range(1, max_episode+1):  
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                st = time.time()
                p = actor.predict_p(s_actor)
                time_cost = time_cost + time.time() - st
                s_actor_next, _, _, _ = env.step(p[:,0])
                s_actor = s_actor_next  
    return time_cost 
    
if __name__ == '__main__':
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    weight_file = 'C:/Software/workshop/python/ddpg_2.mat'
    tf.reset_default_graph()
    with tf.Session() as sess:
        reward_train = Train(sess, env, weight_file)
      
    tf.reset_default_graph()
    with tf.Session() as sess:
        reward_test = Test(sess, env, weight_file)








