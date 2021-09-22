import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras import losses
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import numpy as np
from keras.callbacks import TensorBoard
from keras.utils.generic_utils import get_custom_objects



def VAE_def(intermediate_dim = 32, latent_dim = 2, learning_rate = 0.01, momentum = 0.9):
    
    input_dim = 2
    timesteps = None
    batch_size = 1
    
    epsilon_std=1.

    x = Input(shape=(timesteps, input_dim), name='Encoder_Input')
    
    # LSTM encoding
    h = LSTM(intermediate_dim)(x)
    
    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = tf.random.normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    
    
    
    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)
    
    
    
    def repeat_encoding(input_tensors):
        sequential_input = input_tensors[1]
        to_be_repeated = K.expand_dims(input_tensors[0],axis=1)
    
        # set the one matrix to shape [ batch_size , sequence_length_based on input, 1]
        one_matrix = K.ones_like(sequential_input[:,:,:1])
        
        # do a mat mul
        return K.batch_dot(one_matrix,to_be_repeated)

    # then just call it with a list of tensors 
    h_decoded = Lambda(repeat_encoding,name="repeat_vector_dynamic")([z,x])


    
    
    h_decoded = decoder_h(h_decoded)
    
    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)
    
    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)
    
   
    def vae_loss(x, x_decoded_mean):
        xent_loss = tf.keras.MeanSquaredError(x, x_decoded_mean)
        kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_sigma
                                               - z_mean**2
                                               - tf.exp(z_log_sigma), 1)
        loss = xent_loss + kl_loss
        return loss
    

    
    tf.keras.losses.custom_loss = vae_loss
    opt = Adam(lr=learning_rate,clipnorm=1)
    vae.compile(optimizer=opt, loss=vae_loss)
    

    

    
    return vae


def ENCODER_def(intermediate_dim = 32, latent_dim = 2, learning_rate = 0.01, momentum = 0.9):
    
    input_dim = 2
    timesteps = None
    batch_size = 1
    
    epsilon_std=1.

    x = Input(shape=(timesteps, input_dim), name='Encoder_Input')
    
    # LSTM encoding
    h = LSTM(intermediate_dim)(x)
    
    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = tf.random.normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    
    
    
    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)
    
    # def repeat_vector(args):
    #     layer_to_repeat = args[0]
    #     sequence_layer = args[1]
    #     return RepeatVector(K.shape(sequence_layer)[0])(layer_to_repeat)
    # h_decoded = Lambda(repeat_vector)([z, x])

    def repeat_vector(args):
        sequence_layer = args[1]
        layer_to_repeat = K.expand_dims(args[0],axis=1)
    
        # set the one matrix to shape [ batch_size , sequence_length_based on input, 1]
        one_matrix = K.ones_like(sequence_layer[:,:,:1])
        
        # do a mat mul
        return K.batch_dot(one_matrix,layer_to_repeat)
    
    # then just call it with a list of tensors 
    h_decoded = Lambda(repeat_vector,name="repeat_vector_dynamic")([z,x])

    
    h_decoded = decoder_h(h_decoded)
    
    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)
    
    # end-to-end autoencoder
    encoder = Model(x, z)
    
    
    return encoder


def DECODER_def(intermediate_dim = 32, latent_dim = 2, learning_rate = 0.01, momentum = 0.9):
    
    input_dim = 2
    timesteps = None
    batch_size = 1
    
    epsilon_std=1.

    x = Input(shape=(timesteps, input_dim), name='Encoder_Input')
    
    # LSTM encoding
    h = LSTM(intermediate_dim)(x)
    
    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = tf.random.normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    
    
    
    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)
    
    def repeat_vector(args):
        layer_to_repeat = args[0]
        sequence_layer = args[1]
        return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)
    h_decoded = Lambda(repeat_vector)([z, x])

    
    
    h_decoded = decoder_h(h_decoded)
    
    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)
    
    # end-to-end autoencoder
    decoder = Model(decoder_input, _x_decoded_mean)
    
   

    

    
    return decoder


def train_data_generator(X):
    while True:
        X1 = X
        for i in range(len(X1)):
            a = X1[0]
            
            b = (np.array([a]),np.array([a]))
            yield b
            X1 = X1[1:]
            
def validate_data_generator(X):
    while True:
        X1 = X
        for i in range(len(X1)):
            a = X1[0]
            
            b = (np.array([a]),np.array([a]))
            yield b
            X1 = X1[1:]
            
def test_data_generator(X):
    while True:
        X1 = X
        for i in range(len(X1)):
            a = X1[0]
            
            b = (np.array([a]),np.array([a]))
            yield b
            X1 = X1[1:]
