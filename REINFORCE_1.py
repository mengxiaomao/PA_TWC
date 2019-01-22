# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 14:14:51 2018
REINFORCE
log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=act)
cost = tf.reduce_mean(log_prob * r)
@author: mengxiaomao
"""
import scipy
import numpy as np
import tensorflow as tf
reuse=tf.AUTO_REUSE
   
class DNN:
    def __init__(self, env, weight_file):
        self.state_num = env.state_num
        self.action_num = env.power_num
        self.min_p = 5 #dBm
        self.power_set = env.get_power_set(self.min_p)
        self.M = env.M
        self.weight_file = weight_file

        self.s = tf.placeholder(tf.float32, [None, self.state_num], name ='s')
        self.out = self.create_policy_network(self.s, 'policy')
        self.policy_params = self.get_params('policy')
        self.load_policy_params = self.load_params('policy')
    
    def get_policy_in(self):
        return self.s
            
    def get_policy_out(self):
        return self.out
        
    def get_policy_params(self):
        return self.policy_params
        
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
        
    def create_policy_network(self, s, name):
        with tf.variable_scope(name + '.0', reuse = reuse):
            w = self.variable_w([self.state_num, 128])
            b = self.variable_b([128])
            l = tf.nn.relu(tf.matmul(s, w)+b)
        with tf.variable_scope(name + '.1', reuse = reuse):
            w = self.variable_w([128, 64])
            b = self.variable_b([64])
            l = tf.nn.relu(tf.matmul(l, w) + b)
        with tf.variable_scope(name + '.2', reuse = reuse):
            w = self.variable_w([64, self.action_num])
            b = self.variable_b([self.action_num])
            policy = tf.matmul(l, w) + b
        return policy
        
    def save_params(self):
        dict_name={}
        for var in tf.trainable_variables(): 
            dict_name[var.name]=var.eval()
        scipy.io.savemat(self.weight_file, dict_name)
        
    def load_params(self, name):
        if name == 'policy':
            var_list = self.policy_params
        try:
            theta = scipy.io.loadmat(self.weight_file)
            update=[]
            for var in var_list:
                print(theta[var.name].shape)
                update.append(tf.assign(tf.get_default_graph().get_tensor_by_name(var.name),tf.constant(np.reshape(theta[var.name],var.shape))))
        except:
            print(theta[var.name].shape)
            print('fail')
            update=[]
        return update
        
        
class Policy:
    def __init__(self, sess, dnn, learning_rate = 1e-4):
        self.sess = sess
        self.learning_rate = learning_rate
        self.action_num = dnn.action_num
        self.power_set = dnn.power_set
        self.M = dnn.M
        
        self.r = tf.placeholder(tf.float32, [None])
        self.act = tf.placeholder(tf.int32, [None])
        self.s = dnn.get_policy_in()
        self.out = dnn.get_policy_out()
        self.params = dnn.get_policy_params()
        self.load = dnn.load_policy_params
        self.policy = tf.nn.softmax(self.out, name='act_prob')

        self.log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out, labels=self.act)
        self.cost = tf.reduce_mean(self.log_prob * self.r)

#        self.log_prob = tf.reduce_sum(-tf.log(self.policy)*tf.one_hot(self.act, self.action_num), axis=1)
#        self.cost = tf.reduce_mean(self.log_prob * self.r)
        
        with tf.variable_scope('opt_policy', reuse = reuse):
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost, var_list = self.params)

    def train(self, s, act, r):
        #normalization
        r -= np.mean(r)
        r /= (np.std(r) +1e-8)
        self.sess.run(self.optimize, feed_dict={self.s: s, self.act: act, self.r: r})

    def predict_a(self, s):
        return np.argmax(self.predict_policy(s), axis = 1)
        
    def predict_p(self, s):
        return self.power_set[self.predict_a(s)] 
        
    def predict_policy(self, s):
        return self.sess.run(self.policy, feed_dict={self.s: s})
        
    def load_params(self):
        return self.sess.run(self.load)
        
    def select_action(self, policy):
        act = np.zeros((self.M), dtype = np.int32)
        for i in range(self.M):
            act[i] = np.random.choice(range(self.action_num), p = policy[i,:])
        p = self.power_set[act]
        return p, act
