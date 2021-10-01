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
from keras.callbacks import TensorBoard
from livelossplot import PlotLossesKeras
import skopt




class lstmKerasModel(BaseModel):
    def __init__(self, config, data):
        super(lstmKerasModel, self).__init__(config)
        self.config = config
        # self.intermediate_dim = config['intermediate_dim']
        # self.latent_dim = config['latent_dim']
        # self.learning_rate = config['learning_rate']
        # self.momentum = config['momentum']
        
        self.intermediate_dim = skopt.space.Categorical(categories=[32, 64, 128], name='intermediate_dim')
        self.latent_dim = skopt.space.Integer(low=2, high=5, name='latent_dim')
        self.learning_rate = skopt.space.Categorical(categories=[0.0001,0.001, 0.01, 0.1], name='learning_rate')
        self.momentum = skopt.space.Categorical(categories=[0.2,0.4,0.6,0.8,0.99], name='momentum')
        self.epochs = skopt.space.Integer(low=10, high=100, base = 10, name='epochs')

        # self.intermediate_dim = self.intermediate_dim_range
        # self.latent_dim = self.intermediate_dim_range
        # self.learning_rate = self.learning_rate_range
        # self.momentum = self.momentum_range

        self.dimensions_skopt = [self.intermediate_dim, self.latent_dim, self.learning_rate, self.momentum]
        self.default_parameters = [32, 2, 0.01, 0.9]
        
        self.data = data
        self.build_lstm_model()
        # self.lstm_custom_loss(self.x, self.x_decoded_mean)
        # self.compute_gradients()
        self.produce_embeddings()
        
    
    def build_lstm_model(self):
        self.intermediate_dim = self.config['intermediate_dim']
        self.latent_dim = self.config['latent_dim']
        self.learning_rate = self.config['learning_rate']
        self.momentum = self.config['momentum']
        self.input_dim = self.config['input_dim']
        self.timesteps = None
        self.batch_size = self.config['batch_size']
        self.epsilon_std = self.config['epsilon_std']
      
        self.x = Input(shape=(self.timesteps, self.input_dim), name='Encoder_Input')
        self.h = LSTM(self.intermediate_dim)(self.x)
        self.z_mean = Dense(self.latent_dim)(self.h)
        self.z_log_sigma = Dense(self.latent_dim)(self.h)
      
        def sampling(args):
            self.z_mean, self.z_log_sigma = args
            epsilon = tf.random.normal(shape=(self.batch_size, self.latent_dim),
                                      mean=0., stddev=self.epsilon_std)
            return self.z_mean + self.z_log_sigma * epsilon
      
        self.z = Lambda(sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_sigma])
        self.decoder_h = LSTM(self.intermediate_dim, return_sequences=True)
        self.decoder_mean = LSTM(self.input_dim, return_sequences=True)
      
        def repeat_encoding(input_tensors):
            sequential_input = input_tensors[1]
            to_be_repeated = K.expand_dims(input_tensors[0],axis=1)
            one_matrix = K.ones_like(sequential_input[:,:,:1])
            return K.batch_dot(one_matrix,to_be_repeated)
    
        self.h_decoded = Lambda(repeat_encoding,name="repeat_vector_dynamic")([self.z,self.x])
        self.h_decoded = self.decoder_h(self.h_decoded)
        self.x_decoded_mean = self.decoder_mean(self.h_decoded)
        self.lstm_network = Model(self.x, self.x_decoded_mean)
        
        def lstm_custom_loss(x, x_decoded_mean):
            
            # self.kl_loss = - 0.5 * tf.reduce_sum(1 + self.z_log_sigma
            #                                         - self.z_mean**2
            #                                         - tf.exp(self.z_log_sigma), 1)

            self.xent_loss = tf.keras.metrics.mean_squared_error(x, x_decoded_mean) 
            self.kl_loss =  0.5 * (tf.reduce_sum(tf.square(self.z_mean), 1) + tf.reduce_sum(tf.square(self.z_log_sigma), 1) - tf.reduce_sum(tf.math.log(tf.square(self.z_log_sigma)), 1))
            return self.xent_loss + self.kl_loss
        x = self.x
        x_decoded_mean = self.x_decoded_mean
        # z_mean = self.z_mean
        # z_log_sigma = self.z_log_sigma

        opt = Adam(lr = self.config['learning_rate'],clipnorm= self.config['clipnorm'])
        self.lstm_network.compile(optimizer=opt, loss = lstm_custom_loss)

        
        return self.lstm_network
    
    
    def build_lstm_model2(self, intermediate_dim = 32, latent_dim = 2, learning_rate = 0.01, momentum = 0.9):
        self.input_dim = self.config['input_dim']
        self.timesteps = None
        self.batch_size = self.config['batch_size']
        self.epsilon_std = self.config['epsilon_std']
      
        self.x = Input(shape=(self.timesteps, self.input_dim), name='Encoder_Input')
        self.h = LSTM(intermediate_dim)(self.x)
        self.z_mean = Dense(latent_dim)(self.h)
        self.z_log_sigma = Dense(latent_dim)(self.h)
      
        def sampling(args):
            self.z_mean, self.z_log_sigma = args
            epsilon = tf.random.normal(shape=(self.batch_size, latent_dim),
                                      mean=0., stddev=self.epsilon_std)
            return self.z_mean + self.z_log_sigma * epsilon
      
        self.z = Lambda(sampling, output_shape=(latent_dim,))([self.z_mean, self.z_log_sigma])
        self.decoder_h = LSTM(intermediate_dim, return_sequences=True)
        self.decoder_mean = LSTM(self.input_dim, return_sequences=True)
      
        def repeat_encoding(input_tensors):
            sequential_input = input_tensors[1]
            to_be_repeated = K.expand_dims(input_tensors[0],axis=1)
            one_matrix = K.ones_like(sequential_input[:,:,:1])
            return K.batch_dot(one_matrix,to_be_repeated)
    
        self.h_decoded = Lambda(repeat_encoding,name="repeat_vector_dynamic")([self.z,self.x])
        self.h_decoded = self.decoder_h(self.h_decoded)
        self.x_decoded_mean = self.decoder_mean(self.h_decoded)
        self.lstm_network = Model(self.x, self.x_decoded_mean)
        
        def lstm_custom_loss(x, x_decoded_mean):
            
            # self.kl_loss = - 0.5 * tf.reduce_sum(1 + self.z_log_sigma
            #                                         - self.z_mean**2
            #                                         - tf.exp(self.z_log_sigma), 1)

            self.xent_loss = tf.keras.metrics.mean_squared_error(x, x_decoded_mean) 
            self.kl_loss =  0.5 * (tf.reduce_sum(tf.square(self.z_mean), 1) + tf.reduce_sum(tf.square(self.z_log_sigma), 1) - tf.reduce_sum(tf.math.log(tf.square(self.z_log_sigma)), 1))
            return self.xent_loss + self.kl_loss
        x = self.x
        x_decoded_mean = self.x_decoded_mean
        # z_mean = self.z_mean
        # z_log_sigma = self.z_log_sigma

        opt = Adam(lr = learning_rate,clipnorm= self.config['clipnorm'])
        self.lstm_network.compile(optimizer=opt, loss = lstm_custom_loss)

        
        return self.lstm_network
    
    def build_encoder(self, config):
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
      
        self.z = Lambda(sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_sigma])
        decoder_h = LSTM(self.intermediate_dim, return_sequences=True)
        decoder_mean = LSTM(self.input_dim, return_sequences=True)
      
        def repeat_encoding(input_tensors):
            sequential_input = input_tensors[1]
            to_be_repeated = K.expand_dims(input_tensors[0],axis=1)
            one_matrix = K.ones_like(sequential_input[:,:,:1])
            return K.batch_dot(one_matrix,to_be_repeated)
    
        h_decoded = Lambda(repeat_encoding,name="repeat_vector_dynamic")([self.z,self.x])
        h_decoded = decoder_h(h_decoded)
        self.x_decoded_mean = decoder_mean(h_decoded)
        self.encoder = Model(self.x, self.z)
        
        return self.encoder
    
    
    def build_decoder(self, config):
        self.intermediate_dim = config['intermediate_dim']
        self.latent_dim = config['latent_dim']
        self.learning_rate = config['learning_rate']
        self.momentum = config['momentum']
        self.input_dim = config['input_dim']
        self.timesteps = None
        self.batch_size = config['batch_size']
        self.epsilon_std = config['epsilon_std']
        self.generator_len = config['generator_len']
      
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
        
        self.decoder_input = Input(shape=(self.latent_dim,))
    
        self._h_decoded = repeat_encoding(self.generator_len)(self.decoder_input)
        self._h_decoded2 = decoder_h(self._h_decoded)
        
        self._x_decoded_mean2 = decoder_mean(self._h_decoded2)
        self.decoder = Model(self.decoder_input, self._x_decoded_mean2)
        
        return self.decoder
    
    
    def bayes_opt(self):
        dimensions_skopt = self.dimensions_skopt
        
        @skopt.utils.use_named_args(dimensions = dimensions_skopt)
        def fitness(intermediate_dim, latent_dim, learning_rate, momentum):
            
        
            
            # Create the neural network with these hyper-parameters.
            K.clear_session()
        
            self.lstm_network2 = self.build_lstm_model2(intermediate_dim = self.intermediate_dim, latent_dim = self.latent_dim, learning_rate = self.learning_rate, momentum = self.momentum)
            # Dir-name for the TensorBoard log-files.
            log_dir = './tensorboard_log'
            
            callback_log = TensorBoard(
                log_dir=log_dir,
                histogram_freq=0,
                batch_size=1,
                write_graph=True,
                write_grads=False,
                write_images=False)
            
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
            
            filepath="save_model_skopt/LPT-{epoch:02d}-{loss:.4f}.h5"
            save_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath, save_weights_only=True,monitor='val_loss',save_best_only=False, mode='auto', period=1)
        
            
            # def scheduler(epoch,lr):
                
            #     return lr*(0.99**(epoch//10+1))
                           
                
            # lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler,verbose=1)
            
            class customcallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self,epoch,logs=None):
                    print(logs.keys())
           
            # Use Keras to train the model.
            
        
                        
                        
            
        
            history = self.lstm_network2.fit(self.train_generator,
                                epochs=20,
                                steps_per_epoch=len(self.x_train),
                                validation_data=self.validate_generator,
                                validation_steps=len(self.x_validate),
                                callbacks=[customcallback(), callback_log, save_callback,PlotLossesKeras()])
            accuracy = history.history['val_loss'][-1]
        
        
            del self.lstm_network2
            
            K.clear_session()
            
            return accuracy
        
        checkpoint_saver = skopt.callbacks.CheckpointSaver("save_model_skopt/skopt_saves/checkpoint.pkl", compress=9)

        early_stop1 = skopt.callbacks.EarlyStopper()
        
        fitness_func = fitness
        self.skopt_search_result = skopt.gp_minimize(func=fitness_func,
                                    dimensions=dimensions_skopt,
                                    acq_func='EI', # Expected Improvement.
                                    n_calls=10,
                                    callback=[checkpoint_saver,early_stop1],
                                    random_state=777)
            



    
                  
                  
    def produce_embeddings(self):
        self.x_train = self.data.train_set_lstm
        self.x_test = self.data.test_set_lstm
        self.x_validate = self.data.val_set_lstm
        
        self.train_generator = self.data.data_generator(self.x_train)
        self.train_generator1 = self.data.test_data_generator(self.x_train)

        self.test_generator = self.data.test_data_generator(self.x_test)
        self.validate_generator = self.data.data_generator(self.x_validate)



    def load_saved_model(self, lstm_network, config, checkpoint_path):
        print(config['checkpoint_dir_lstm'] + 'checkpoint')
        if os.path.isfile(config['checkpoint_dir_lstm'] + 'checkpoint'):
            lstm_network.load_weights(checkpoint_path)
            print("LSTM model loaded.")
        else:
            print("No LSTM model loaded.")
            
    
    def train(self, config, lstm_network, cp_callback):
        lstm_network.fit(self.train_generator,
                        epochs = self.config['num_epochs_lstm'],
                        steps_per_epoch = len(self.x_train),
                        validation_data = self.validate_generator,
                        validation_steps=len(self.x_validate),
                        callbacks=[cp_callback])
        
    def test(self, lstm_network):
        lstm_network.predict(next(self.test_generator))

      
      
      
     