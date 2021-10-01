#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:55:33 2021

@author: hossam
"""

from base import BaseDataGenerator
import numpy as np


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
                  
    def test_data_generator(self,X):
        while True:
            X1 = X
            for i in range(len(X1)):
                a = X1[0]
                
                b = (np.array([a]))
                yield b
                X1 = X1[1:]


        