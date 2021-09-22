from lstm_vae.read_data import read_data, arr2df, TrajVar, TrajVar_no_smooth, Traj_df_var_filter, Traj_arr_filtered, data_split
from lstm_vae.network import VAE_def

def main():
    Traj_arr = read_data('data/Burrard_unconstrained.csv')
    Traj_df = arr2df(Traj_arr)
    Traj_df_var = TrajVar(Traj_df, Traj_arr)
    Traj_df_var_no_smooth = TrajVar_no_smooth(Traj_df, Traj_arr)
    Traj_df_var_filtered = Traj_df_var_filter(Traj_df_var)
    Traj_arr_filtered1 = Traj_arr_filtered(Traj_df_var_filtered)
    X_train, X_test = data_split(Traj_arr_filtered1, random_state = 333, test_size = 0.2)
    X_train1, X_validate = data_split(X_train, random_state = 333, test_size = 0.33)

    for i in range(len(X_train1)):
        for j in range(len(X_train1[i])):
        
            X_train1[i][j][0] = (X_train1[i][j][0]-6.5) /(9-6.5)
            X_train1[i][j][1] = (X_train1[i][j][1]-1)/(37.33-1)
        
        
    for i in range(len(X_validate)):
        for j in range(len(X_validate[i])):
            
            X_validate[i][j][0] = (X_validate[i][j][0]-6.5) /(9-6.5)
            X_validate[i][j][1] = (X_validate[i][j][1]-1)/(37.33-1)

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
