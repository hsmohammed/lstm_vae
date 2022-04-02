# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 12:36:59 2022

@author: hossameldin_mohammed
"""

from data_loader import DataGenerator
from models import lstmKerasModel
from utils import  create_dirs, get_args, process_config
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import interpolate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.ticker import FormatStrFormatter
import pickle


sns.set(font_scale = 2, style='whitegrid')

config = process_config("unconstrained_config.json")
create_dirs([config['visualization']])


data = DataGenerator(config)
lstm_model = lstmKerasModel(config, data)
# lstm_model.compute_gradients()

lstm_network = lstm_model.build_lstm_model()
encoder_model = lstm_model.build_encoder(config)
# decoder_model = lstm_model.build_decoder(config)

checkpoint_path = config['checkpoint_dir_lstm'] + "cp.ckpt"
lstm_model.load_saved_model(lstm_network, config, checkpoint_path)
# lstm_weights = lstm_nn_model.get_weights()

# encoder_weights = lstm_weights[:7]
# decoder_weights = lstm_weights[7:]
# encoder_model.set_weights(encoder_weights)
# decoder_model.set_weights(decoder_weights)
test_generator = lstm_model.data.test_data_generator(lstm_model.x_test)