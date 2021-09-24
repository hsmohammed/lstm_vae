#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 12:14:40 2021

@author: hossam
"""
import tensorflow as tf
import numpy as np
# from utils import count_trainable_variables
import time
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, savefig, figure
from tensorflow.keras.optimizers import Adam

class BaseDataGenerator:
  def __init__(self, config):
    self.config = config



class BaseModel:
    def __init__(self, config):
        self.config = config
        self.two_pi = tf.constant(2*np.pi)
        
    def lstm_loss(self, x, x_decoded_mean):
        # KL divergence loss - analytical result
        xent_loss = tf.keras.metrics.mean_squared_error(x, x_decoded_mean)
        # kl_loss = - 0.5 * tf.reduce_sum(1 + self.z_log_sigma - self.z_mean**2 - tf.exp(self.z_log_sigma), 1)
        self.loss = xent_loss
        return self.loss
    
    
    def compute_gradients(self):
        tf.keras.losses.custom_loss = self.lstm_loss
        opt = Adam(lr = self.config['learning_rate'],clipnorm= self.config['clipnorm'])
        self.lstm_model.compile(optimizer=opt, loss = self.lstm_loss)


        
    # save function that saves the checkpoint in the path defined in the config file
    # def save(self, sess):
    #     print("Saving model...")
    #     cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = self.config['checkpoint_dir'],
    #                                                save_weights_only = True,
    #                                                verbose=1)
    #     print("Model saved.")
      
    # # load latest checkpoint from the experiment path defined in the config file
    # def load(self, sess):
    #     print("checkpoint_dir at loading: {}".format(self.config['checkpoint_dir']))
    #     self.load_weights(self.config['checkpoint_dir'])
    
    #     print("Model loaded.")
      
  
    # def define_loss(self):
    #     # KL divergence loss - analytical result
    #     xent_loss = tf.keras.metrics.mean_squared_error(self.x, self.x_decoded_mean)
    #     # kl_loss = - 0.5 * tf.reduce_sum(1 + self.z_log_sigma - self.z_mean**2 - tf.exp(self.z_log_sigma), 1)
    #     self.loss = xent_loss 
    

  
    # def compute_gradients(self):
    #     # tf.keras.losses.custom_loss = self.define_loss
    #     opt = Adam(lr = self.config['learning_rate'],clipnorm= self.config['clipnorm'])
    #     self.lstm_model.compile(optimizer=opt, loss = self.define_loss)
      
# class BaseTrain:
#   def __init__(self, sess, model, data, config):
#     self.model = model
#     self.config = config
#     self.data = data

#     # keep a record of the training result
#     self.train_loss = []
#     self.val_loss = []
#     self.train_loss_ave_epoch = []
#     self.val_loss_ave_epoch = []
#     self.sample_std_dev_train = []
#     self.sample_std_dev_val = []
#     self.iter_epochs_list = []
#     self.test_sigma2 = []

#   def train(self):
#     self.start_time = time.time()
#     for cur_epoch in range(0, self.config['num_epochs_vae'], 1):
#       self.train_epoch()

#       # compute current execution time
#       self.current_time = time.time()
#       elapsed_time = (self.current_time - self.start_time) / 60
#       est_remaining_time = (
#                                    self.current_time - self.start_time) / (cur_epoch + 1) * (
#                                      self.config['num_epochs_vae'] - cur_epoch - 1)
#       est_remaining_time = est_remaining_time / 60
#       print("Already trained for {} min; Remaining {} min.".format(elapsed_time, est_remaining_time))

#   def save_variables_VAE(self):
#     # save some variables for later inspection
#     file_name = "{}{}-batch-{}-epoch-{}-code-{}-lr-{}.npz".format(self.config['result_dir'],
#                                                                   self.config['exp_name'],
#                                                                   self.config['batch_size'],
#                                                                   self.config['num_epochs_vae'],
#                                                                   self.config['code_size'],
#                                                                   self.config['learning_rate_vae'])
#     np.savez(file_name,
#              iter_list_val=self.iter_epochs_list,
#              train_loss=self.train_loss,
#              val_loss=self.val_loss,
#              n_train_iter=self.n_train_iter,
#              sigma2=self.test_sigma2)

#   def plot_train_and_val_loss(self):
#     # plot the training and validation loss over epochs
#     plt.clf()
#     figure(num=1, figsize=(8, 6))
#     plot(self.train_loss, 'b-')
#     plot(self.iter_epochs_list, self.val_loss_ave_epoch, 'r-')
#     plt.legend(('training loss (total)', 'validation loss'))
#     plt.title('training loss over iterations (val @ epochs)')
#     plt.ylabel('total loss')
#     plt.xlabel('iterations')
#     plt.grid(True)
#     savefig(self.config['result_dir'] + '/loss.png')


#     # plot individual components of validation loss over epochs
#     plt.clf()
#     figure(num=1, figsize=(8, 6))
#     plot(self.test_sigma2, 'b-')
#     plt.title('sigma2 over training')
#     plt.ylabel('sigma2')
#     plt.xlabel('iter')
#     plt.grid(True)
#     savefig(self.config['result_dir'] + '/sigma2.png')
  
    
    
