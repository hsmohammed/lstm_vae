#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:08:18 2021

@author: hossam
"""
from base import BaseModel
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
import os
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam




class lstmKerasModel(BaseModel):
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.build_lstm_model(self.config)
        self.produce_embeddings()
        self.lstm_loss(self.x, self.x_decoded_mean)
        self.compute_gradients()
    
    def build_lstm_model(self, config):
        self.intermediate_dim = config['intermediate_dim']
        self.latent_dim = config['latent_dim']
        self.learning_rate = config['learning_rate']
        self.momentum = config['momentum']
        self.input_dim = config['input_dim']
        self.timesteps = None
        self.batch_size = config['batch_size']
        self.epsilon_std = config['epsilon_std']
      
        self.x = Input(shape=(self.timesteps, self.input_dim), name='Encoder_Input')
        h = LSTM(self.intermediate_dim)(self.x)
        self.z_mean = Dense(self.latent_dim)(h)
        self.z_log_sigma = Dense(self.latent_dim)(h)
      
        def sampling(args):
            self.z_mean, self.z_log_sigma = args
            epsilon = tf.random.normal(shape=(self.batch_size, self.latent_dim),
                                      mean=0., stddev=self.epsilon_std)
            return self.z_mean + self.z_log_sigma * epsilon
      
        z = Lambda(sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_sigma])
        decoder_h = LSTM(self.intermediate_dim, return_sequences=True)
        decoder_mean = LSTM(self.input_dim, return_sequences=True)
      
        def repeat_encoding(input_tensors):
            sequential_input = input_tensors[1]
            to_be_repeated = K.expand_dims(input_tensors[0],axis=1)
            one_matrix = K.ones_like(sequential_input[:,:,:1])
            return K.batch_dot(one_matrix,to_be_repeated)
    
        h_decoded = Lambda(repeat_encoding,name="repeat_vector_dynamic")([z,self.x])
        h_decoded = decoder_h(h_decoded)
        self.x_decoded_mean = decoder_mean(h_decoded)
        self.lstm_model = Model(self.x, self.x_decoded_mean)
        
        


        
        return self.lstm_model
    
                  
                  
    def produce_embeddings(self):
        self.x_train = self.data.train_set_lstm
        self.x_test = self.data.test_set_lstm
        self.x_validate = self.data.val_set_lstm
        
        self.train_generator = self.data.data_generator(self.x_train)
        self.test_generator = self.data.data_generator(self.x_test)
        self.validate_generator = self.data.data_generator(self.x_validate)



    def load_model(self, lstm_model, config, checkpoint_path):
        print(config['checkpoint_dir_lstm'] + 'checkpoint')
        if os.path.isfile(config['checkpoint_dir_lstm'] + 'checkpoint'):
            lstm_model.load_weights(checkpoint_path)
            print("LSTM model loaded.")
        else:
            print("No LSTM model loaded.")
            
    
    def train(self, config, lstm_model, cp_callback):
        lstm_model.fit(self.train_generator,
                        epochs = self.config['num_epochs_lstm'],
                        steps_per_epoch = len(self.x_train),
                        validation_data = self.validate_generator,
                        validation_steps=len(self.x_validate),
                        callbacks=[cp_callback])

      
      
      
      
      
      
    
    # def plot_reconstructed_lt_seq(self, idx_test, config, model_vae, sess, data, lstm_embedding_test):
    #   feed_dict_vae = {model_vae.original_signal: np.zeros((config['l_seq'], config['l_win'], config['n_channel'])),
    #                    model_vae.is_code_input: True,
    #                    model_vae.code_input: self.embedding_lstm_test[idx_test]}
    #   decoded_seq_vae = np.squeeze(sess.run(model_vae.decoded, feed_dict=feed_dict_vae))
    #   print("Decoded seq from VAE: {}".format(decoded_seq_vae.shape))
    
    #   feed_dict_lstm = {model_vae.original_signal: np.zeros((config['l_seq'] - 1, config['l_win'], config['n_channel'])),
    #                     model_vae.is_code_input: True,
    #                     model_vae.code_input: lstm_embedding_test[idx_test]}
    #   decoded_seq_lstm = np.squeeze(sess.run(model_vae.decoded, feed_dict=feed_dict_lstm))
    #   print("Decoded seq from lstm: {}".format(decoded_seq_lstm.shape))
    
    #   fig, axs = plt.subplots(config['n_channel'], 2, figsize=(15, 4.5 * config['n_channel']), edgecolor='k')
    #   fig.subplots_adjust(hspace=.4, wspace=.4)
    #   axs = axs.ravel()
    #   for j in range(config['n_channel']):
    #     for i in range(2):
    #       axs[i + j * 2].plot(np.arange(0, config['l_seq'] * config['l_win']),
    #                           np.reshape(data.val_set_lstm['data'][idx_test, :, :, j],
    #                                      (config['l_seq'] * config['l_win'])))
    #       axs[i + j * 2].grid(True)
    #       axs[i + j * 2].set_xlim(0, config['l_seq'] * config['l_win'])
    #       axs[i + j * 2].set_xlabel('samples')
    #     if config['n_channel'] == 1:
    #       axs[0 + j * 2].plot(np.arange(0, config['l_seq'] * config['l_win']),
    #                           np.reshape(decoded_seq_vae, (config['l_seq'] * config['l_win'])), 'r--')
    #       axs[1 + j * 2].plot(np.arange(config['l_win'], config['l_seq'] * config['l_win']),
    #                           np.reshape(decoded_seq_lstm, ((config['l_seq'] - 1) * config['l_win'])), 'g--')
    #     else:
    #       axs[0 + j * 2].plot(np.arange(0, config['l_seq'] * config['l_win']),
    #                           np.reshape(decoded_seq_vae[:, :, j], (config['l_seq'] * config['l_win'])), 'r--')
    #       axs[1 + j * 2].plot(np.arange(config['l_win'], config['l_seq'] * config['l_win']),
    #                           np.reshape(decoded_seq_lstm[:, :, j], ((config['l_seq'] - 1) * config['l_win'])), 'g--')
    #     axs[0 + j * 2].set_title('VAE reconstruction - channel {}'.format(j))
    #     axs[1 + j * 2].set_title('LSTM reconstruction - channel {}'.format(j))
    #     for i in range(2):
    #       axs[i + j * 2].legend(('ground truth', 'reconstruction'))
    #     savefig(config['result_dir'] + "lstm_long_seq_recons_{}.pdf".format(idx_test))
    #     fig.clf()
    #     plt.close()
    
    # def plot_lstm_embedding_prediction(self, idx_test, config, model_vae, sess, data, lstm_embedding_test):
    #   self.plot_reconstructed_lt_seq(idx_test, config, model_vae, sess, data, lstm_embedding_test)
    
    #   fig, axs = plt.subplots(2, config['code_size'] // 2, figsize=(15, 5.5), edgecolor='k')
    #   fig.subplots_adjust(hspace=.4, wspace=.4)
    #   axs = axs.ravel()
    #   for i in range(config['code_size']):
    #     axs[i].plot(np.arange(1, config['l_seq']), np.squeeze(self.embedding_lstm_test[idx_test, 1:, i]))
    #     axs[i].plot(np.arange(1, config['l_seq']), np.squeeze(lstm_embedding_test[idx_test, :, i]))
    #     axs[i].set_xlim(1, config['l_seq'] - 1)
    #     axs[i].set_ylim(-2.5, 2.5)
    #     axs[i].grid(True)
    #     axs[i].set_title('Embedding dim {}'.format(i))
    #     axs[i].set_xlabel('windows')
    #     if i == config['code_size'] - 1:
    #       axs[i].legend(('VAE\nembedding', 'LSTM\nembedding'))
    #   savefig(config['result_dir'] + "lstm_seq_embedding_{}.pdf".format(idx_test))
    #   fig.clf()
    #   plt.close()