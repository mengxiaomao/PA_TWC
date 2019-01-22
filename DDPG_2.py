# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:54:26 2018
actor-critic 
end-to-end DDPG
@author: mengxiaomao
"""
import scipy
import numpy as np
import tensorflow as tf
reuse=tf.AUTO_REUSE
dtype = np.float32

class DDPG:
    def __init__(self, env, weight_file):
        self.weight_file = weight_file
        
        self.state_actor_num = env.state_num
        self.action_num = env.power_num
        self.a_bound = env.maxP
        self.sigma2 = env.sigma2
        self.p_list = env.p_list
        self.M = env.M
        self.C = env.C
        self.state_critic_num = env.K
        
        self.s_actor = tf.placeholder(tf.float32, [None, self.state_actor_num], name ='actor_input_s')
        self.p_actor = self.create_actor(self.s_actor, 'actor')
        self.actor_params = self.get_params('actor')
        self.load_actor_params = self.load_params('actor')

        self.s_critic = tf.placeholder(tf.float32, [None, self.state_critic_num], name ='critic_input_s')
        self.p_critic = tf.placeholder(tf.float32, [None, self.action_num], name ='critic_input_a')
        self.critic = self.create_critic(self.s_critic, self.p_critic, 'critic')
        self.critic_params = self.get_params('critic')
        self.load_critic_params = self.load_params('critic')

    def get_actor_in(self):
        return self.s_actor
    
    def get_actor_out(self):
        return self.p_actor
        
    def get_actor_params(self):
        return self.actor_params  
    
    def get_critic_in(self):
        return (self.s_critic, self.p_critic)
    
    def get_critic_out(self):
        return self.critic
        
    def get_critic_params(self):
        return self.critic_params
        
    def get_params(self, para_name):
        sets=[]
        for var in tf.trainable_variables():
            if not var.name.find(para_name):
                sets.append(var)
        return sets
        
    def variable_w(self, shape, name = 'w'):
        w = tf.get_variable(name, shape = shape, initializer = tf.truncated_normal_initializer(stddev=0.1))
        return w
        
    def variable_b(self, shape, initial = 0.01):
        b = tf.get_variable('b', shape = shape, initializer = tf.constant_initializer(initial))    
        return b  
        
    def create_critic(self, s_critic, p_critic, name):
        '''
        s_critic: [M, K]
        p_critic: [M,1]
        rate_matrix [M, K]
        '''
        maxC = 1000.
        P_extend = tf.concat([p_critic[:,0], tf.zeros((1), dtype = dtype)], axis=0)
        P_matrix = tf.gather_nd(P_extend, self.p_list)
        path_main = tf.multiply(s_critic[:,0], P_matrix[:,0])
        path_inter = tf.reduce_sum(tf.multiply(s_critic[:,1:], P_matrix[:,1:]), axis=1)
        sinr = tf.minimum(path_main / (path_inter + self.sigma2), maxC)
        rate = tf.log(1. + sinr)/tf.log(2.)
        rate_extend = tf.concat([rate, tf.zeros((1), dtype = dtype)], axis=0)
        rate_matrix = tf.gather_nd(rate_extend, self.p_list)
        rate_matrix = tf.nn.top_k(rate_matrix, self.C)[0]

        with tf.variable_scope(name + '.0', reuse = reuse):
            w = self.variable_w([self.C, 64])
            b = self.variable_b([64])
            l = tf.nn.relu(tf.matmul(rate_matrix, w)+b)
        with tf.variable_scope(name + '.1', reuse = reuse):
            w = self.variable_w([64, 1])
            b = self.variable_b([1], initial = 0.0)
            l = tf.matmul(l, w) + b
            q_hat = tf.reduce_sum(l, axis = 1)
        return q_hat

#    def create_critic(self, s_critic, p_critic, name):
#        '''
#        s_critic: [M, K]
#        p_critic: [M,1]
#        rate_matrix [M, K]
#        '''
#        maxC = 1000.
#        P_extend = tf.concat([p_critic[:,0], tf.zeros((1), dtype = dtype)], axis=0)
#        P_matrix = tf.gather_nd(P_extend, self.p_list)
#        path_main = tf.multiply(s_critic[:,0], P_matrix[:,0])
#        path_inter = tf.reduce_sum(tf.multiply(s_critic[:,1:], P_matrix[:,1:]), axis=1)
#        sinr = tf.minimum(path_main / (path_inter + self.sigma2), maxC)
#        rate = tf.log(1. + sinr)/tf.log(2.)
#        rate_extend = tf.concat([rate, tf.zeros((1), dtype = dtype)], axis=0)
#        rate_matrix = tf.gather_nd(rate_extend, self.p_list)
#        q_hat = rate + tf.reduce_sum(rate_matrix, axis=1)
#        return q_hat
        
    def create_actor(self, s_actor, name):
        with tf.variable_scope(name + '.0', reuse = reuse):
            w = self.variable_w([self.state_actor_num, 128])
            b = self.variable_b([128])
            l = tf.nn.relu(tf.matmul(s_actor, w)+b)
        with tf.variable_scope(name + '.1', reuse = reuse):
            w = self.variable_w([128, 64])
            b = self.variable_b([64])
            l = tf.nn.relu(tf.matmul(l, w)+b)
        with tf.variable_scope(name + '.2', reuse = reuse):
            w = self.variable_w([64, self.action_num])
            b = self.variable_b([self.action_num])
            l = tf.nn.sigmoid(tf.matmul(l, w) + b)
        with tf.variable_scope(name + '.norm', reuse = reuse):
            a_hat = l * self.a_bound
        return a_hat
        
    def save_params(self):
        dict_name={}
        for var in tf.trainable_variables(): 
            dict_name[var.name]=var.eval()
        scipy.io.savemat(self.weight_file, dict_name)
        
    def load_params(self, name):
        if name == 'actor':
            var_list = self.actor_params         
        elif name == 'critic':
            var_list = self.critic_params
        try:
            
            theta = scipy.io.loadmat(self.weight_file)
            print(theta)
            update=[]
            for var in var_list:
                update.append(tf.assign(tf.get_default_graph().get_tensor_by_name(var.name),tf.constant(np.reshape(theta[var.name],var.shape))))
        except:
            print('fail ddpg')
            update=[]
        return update

        
class Actor():    
    def __init__(self, sess, dnn, learning_rate = 1e-4):
        self.sess = sess
        self.learning_rate = learning_rate
        self.action_num = dnn.action_num
        self.maxP = dnn.a_bound
        self.M = dnn.M
        
        self.std = tf.placeholder(tf.float32, name = 'std')
#        self.noise = tf.placeholder(tf.float32, [None, self.action_num])
        self.s_actor = dnn.get_actor_in()
        self.p_actor = dnn.get_actor_out()
        self.p_exp = self.select_action()
        self.params = dnn.get_actor_params()
        self.load = dnn.load_actor_params

        
        self.critic_gradient = tf.placeholder(tf.float32, [None, self.action_num])
        self.policy_gradient = tf.gradients(tf.multiply(self.p_actor, -self.critic_gradient), self.params)     
        with tf.variable_scope('opt_actor', reuse = reuse):
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
                apply_gradients(zip(self.policy_gradient, self.params)) 
                
    def train(self, s_actor, critic_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.s_actor: s_actor, self.critic_gradient: critic_gradient})

    def predict_p(self, s_actor):
        return self.sess.run(self.p_actor, feed_dict={self.s_actor: s_actor})
    
    def load_params(self):
        return self.sess.run(self.load)
        
    def select_action(self):
#        noise = tf.random_normal(shape = (self.M, self.action_num), stddev=1e-2)
#        print(tf.shape(self.p_actor)[0])
#        print(self.p_actor.shape[0])
        noise = tf.random_uniform(shape = (self.M, self.action_num), minval=-self.maxP/self.std, maxval=self.maxP/self.std)
        act = tf.minimum(self.maxP, tf.maximum(0., (self.p_actor + noise)))
        return act
        
    def get_random_action(self, s_actor, std):
        return self.sess.run([self.p_actor, self.p_exp], feed_dict={self.s_actor: s_actor, self.std: std})
     
        
class Critic:
    def __init__(self, sess, dnn, learning_rate = 1e-3):
        self.sess = sess
        self.learning_rate = learning_rate

        self.q_tar = tf.placeholder(tf.float32, [None])
        
        self.s_critic, self.p_critic = dnn.get_critic_in()
        self.q = dnn.get_critic_out()
        self.params = dnn.get_critic_params()
        self.load = dnn.load_critic_params

        self.action_grads = tf.gradients(self.q, self.p_critic)
        self.cost = tf.nn.l2_loss(self.q_tar - self.q)
        with tf.variable_scope('opt_critic', reuse = reuse):
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def train(self, s_critic, p_critic, q_tar):
        self.sess.run(self.optimize, feed_dict={self.s_critic: s_critic, self.p_critic: p_critic, self.q_tar: q_tar})
        return self.sess.run(self.cost, feed_dict={self.s_critic: s_critic, self.p_critic: p_critic, self.q_tar: q_tar})
    
    def loss(self, s_critic, p_critic, q_tar):
        return self.sess.run(self.cost, feed_dict={self.s_critic: s_critic, self.p_critic: p_critic, self.q_tar: q_tar})

    def predict_q(self, s_critic, p_critic):
        return self.sess.run(self.q, feed_dict={self.s_critic: s_critic, self.p_critic: p_critic})
        
    def get_gradient(self, s_critic, p_critic):
        return self.sess.run(self.action_grads, feed_dict={self.s_critic: s_critic, self.p_critic: p_critic})
        
    def load_params(self):
        return self.sess.run(self.load)

        
     
           