import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, ion, show, savefig, cla, figure
import random
import time
import lstm_vae

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# from data_loader import DataGenerator
from models import lstmKerasModel
from trainers import vaeTrainer

from utils import process_config, create_dirs, get_args

# from tensorflow.compat.v1.python.client import device_lib


# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']

# print(get_available_gpus())


def main():


    with np.load('data/processed_data.npz', allow_pickle=True) as processed_data:
        X_train = processed_data['X_train']
        X_test = processed_data['X_test']
        X_validate = processed_data['X_validate']
    
    
    config = process_config('src/NAB_config.json')
    # create the experiments dirs
    create_dirs([config['result_dir'], config['checkpoint_dir']])
    # create tensorflow session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # create your data generator
    data = X_train
    # create a CNN model
    model_vae = lstmKerasModel(config)
    # create a CNN model
    trainer_vae = vaeTrainer(sess, model_vae, data, config)
    model_vae.load(sess)
    
    lstm_model = lstmKerasModel(data)
    lstm_model.produce_embeddings(config, model_vae, data, sess)
    lstm_nn_model = lstm_model.create_lstm_model(config)
    lstm_nn_model.summary()   # Display the model's architecture
    
    # checkpoint path
    checkpoint_path = config['checkpoint_dir_lstm'] + "cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    # load weights if possible
    lstm_model.load_model(lstm_nn_model, config, checkpoint_path)


if __name__ == "__main__":
    main()
