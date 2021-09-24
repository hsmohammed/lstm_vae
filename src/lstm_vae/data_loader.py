#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:55:33 2021

@author: hossam
"""

from base import BaseDataGenerator
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig


class DataGenerator(BaseDataGenerator):
    def __init__(self, config):
        super(DataGenerator, self).__init__(config)
        # load data here: generate 3 state variables: train_set, val_set and test_set
        self.load_dataset(self.config['data_dir'])

    def load_dataset(self, dataset):
    
        with np.load(dataset, allow_pickle=True) as processed_data:
            self.train_set_lstm = processed_data['X_train']
            self.test_set_lstm = processed_data['X_test']
            self.val_set_lstm = processed_data['X_validate']
            
    def data_generator(self,X):
          while True:
              X1 = X
              for i in range(len(X1)):
                  a = X1[0]
                  
                  b = (np.array([a]),np.array([a]))
                  yield b
                  X1 = X1[1:]

        

        
  




  # def plot_time_series(self, data, time, data_list):
  #   fig, axs = plt.subplots(1, 4, figsize=(18, 2.5), edgecolor='k')
  #   fig.subplots_adjust(hspace=.8, wspace=.4)
  #   axs = axs.ravel()
  #   for i in range(4):
  #     axs[i].plot(time / 60., data[:, i])
  #     axs[i].set_title(data_list[i])
  #     axs[i].set_xlabel('time (h)')
  #     axs[i].set_xlim((np.amin(time) / 60., np.amax(time) / 60.))
  #   savefig(self.config['result_dir'] + '/raw_training_set_normalised.pdf')