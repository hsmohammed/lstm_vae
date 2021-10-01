#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:11:39 2021

@author: hossam
"""
import tensorflow as tf
from data_loader import DataGenerator
from models import lstmKerasModel
from utils import process_config, create_dirs, get_args, save_config
import os

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    # try:
    #     args = get_args()
    #     # config = process_config(args.config)
    # except:
    #     print("missing or invalid arguments")
    #     exit(0)
    args = get_args()
    config = process_config(args.config)
    # config = process_config("unconstrained_config.json")
    # create the experiments dirs
    create_dirs([config['result_dir'], config['checkpoint_dir'], config['checkpoint_dir_lstm']])
    # save the config in a txt file
    save_config(config)
    # create your data generator
    # tf.config.run_functions_eagerly(False)
    # tf.compat.v1.enable_eager_execution()

    data = DataGenerator(config)

    # create a lstm model class instance
    lstm_model = lstmKerasModel(config, data)


    # Create a basic model instance
    lstm_network = lstm_model.lstm_network
    lstm_network.summary()   # Display the model's architecture
    # checkpoint path
    checkpoint_path = config['checkpoint_dir_lstm']\
                      + "cp.ckpt"
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)
    # load weights if possible
    # lstm_model.load_saved_model(lstm_network, config, checkpoint_path)

    # start training
    if config['num_epochs_lstm'] > 0:
        lstm_model.train(config, lstm_network, cp_callback)

    # # make a prediction on the test set using the trained model
    # lstm_test = lstm_model.test(lstm_nn_model)
    # print(lstm_test.shape)



if __name__ == '__main__':
    main()
