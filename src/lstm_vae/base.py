#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 12:14:40 2021

@author: hossam
"""
import tensorflow as tf
import numpy as np
# from utils import count_trainable_variables
from tensorflow.keras.optimizers import Adam

class BaseDataGenerator:
  def __init__(self, config):
    self.config = config



class BaseModel:
    def __init__(self, config):
        self.config = config
        self.two_pi = tf.constant(2*np.pi)
        
    
    # def lstm_custom_loss(self, x, x_decoded_mean, z_mean, z_log_sigma):
        
    #     # self.kl_loss = - 0.5 * tf.reduce_sum(1 + self.z_log_sigma
    #     #                                         - self.z_mean**2
    #     #                                         - tf.exp(self.z_log_sigma), 1)

    #     self.xent_loss = tf.keras.metrics.mean_squared_error(x, x_decoded_mean) 
    #     self.kl_loss =  0.5 * (tf.reduce_sum(tf.square(self.z_mean), 1) + tf.reduce_sum(tf.square(self.z_log_sigma), 1) - tf.reduce_sum(tf.math.log(tf.square(self.z_log_sigma)), 1))
    #     return self.xent_loss + self.kl_loss

    
    # def compute_gradients(self):
    #     x = self.x
    #     x_decoded_mean = self.x_decoded_mean
    #     z_mean = self.z_mean
    #     z_log_sigma = self.z_log_sigma
    #     opt = Adam(lr = self.config['learning_rate'],clipnorm= self.config['clipnorm'])
    #     self.lstm_network.compile(optimizer=opt, loss = self.lstm_custom_loss)

