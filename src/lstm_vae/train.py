#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:11:39 2021

@author: hossam
"""
import os
import tensorflow as tf
from data_loader import DataGenerator
from models import lstmKerasModel
# from trainers import vaeTrainer
from utils import process_config, create_dirs, get_args, save_config


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
    # create the experiments dirs
    create_dirs([config['result_dir'], config['checkpoint_dir'], config['checkpoint_dir_lstm']])
    # save the config in a txt file
    save_config(config)
    # create your data generator
    tf.config.experimental_run_functions_eagerly(True)
    data = DataGenerator(config)

    # create a lstm model class instance
    lstm_model = lstmKerasModel(config, data)


    # Create a basic model instance
    lstm_nn_model = lstm_model.build_lstm_model(config)
    lstm_nn_model.summary()   # Display the model's architecture
    # checkpoint path
    checkpoint_path = config['checkpoint_dir_lstm']\
                      + "cp.ckpt"
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)
    # load weights if possible
    lstm_model.load_model(lstm_nn_model, config, checkpoint_path)

    # start training
    if config['num_epochs_lstm'] > 0:
        lstm_model.compute_gradients()
        lstm_model.train(config, lstm_nn_model, cp_callback)

    # make a prediction on the test set using the trained model
    lstm_embedding = lstm_nn_model.predict(next(lstm_nn_model.test_generator))
    print(lstm_embedding.shape)

    # visualise the first 10 test sequences
    # for i in range(10):
    #     lstm_model.plot_lstm_embedding_prediction(i, config, model_vae, sess, data, lstm_embedding)


if __name__ == '__main__':
    main()
