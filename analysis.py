from lstm_vae.read_data import read_data, arr2df, TrajVar, TrajVar_no_smooth, Traj_df_var_filter, Traj_arr_filtered, data_split
from lstm_vae.network import VAE_def
import numpy as np

def main():
    
    with np.load('data/processed_data.npz', allow_pickle=True) as processed_data:
        X_train = processed_data['X_train']
        X_test = processed_data['X_test']
        X_validate = processed_data['X_validate']




    # intermediate_dim = Categorical(categories=[32, 64, 128],
    #                      name='intermediate_dim')


    # latent_dim = Integer(low=2, high=5, name='latent_dim')

    # learning_rate = Categorical(categories=[0.0001,0.001, 0.01, 0.1], name='learning_rate')

    # momentum = Categorical(categories=[0.2,0.4,0.6,0.8,0.99], name='momentum')

    # epochs = Integer(low=10, high=100, base = 10, name='epochs')

    # dimensions = [intermediate_dim,
    #               latent_dim,
    #               learning_rate,
    #               momentum]

    # default_parameters = [32, 2, 0.01, 0.9]


    model = VAE_def(intermediate_dim = 5, latent_dim = 64, learning_rate = 0.0001, momentum = 0)

    model.load_weights('model_weights/LPT-429.h5')



    model.summary()


if __name__ == "__main__":
    main()
