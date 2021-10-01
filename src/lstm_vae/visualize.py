#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:38:02 2021

@author: hossam
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

def main():
    # args = get_args()
    # config = process_config(args.config)

    sns.set(font_scale = 2, style='whitegrid')

    config = process_config("unconstrained_config.json")
    create_dirs([config['visualization']])


    data = DataGenerator(config)
    lstm_model = lstmKerasModel(config, data)
    # lstm_model.compute_gradients()

    lstm_network = lstm_model.lstm_model
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
    
    # tested = lstm_network.predict(next(test_generator))

    # tested.shape
    
    
    # lstm_test = lstm_model.test(lstm_nn_model)
    # print(lstm_test.shape)
    
    # # for i in range(len(X_test)):
    # #     for j in range(len(X_test[i])):
        
    # #         X_test[i][j][0] = (X_test[i][j][0]-6.5) /(9-6.5)
    # #         X_test[i][j][1] = (X_test[i][j][1]-1)/(37.33-1)
    
    
    X_test_pred = []
    
    for i in range(len(lstm_model.x_test)):
        
        X_test_pred_current = lstm_network.predict(next(test_generator))
        X_test_pred.append(X_test_pred_current)
    
    
        
        
    RMSE = []
    for i in range(len(lstm_model.x_test)):
        rms = sqrt(mean_squared_error(lstm_model.x_test[i], X_test_pred[i][0]))
        RMSE.append(rms)
        
    
    
    
    
    # rms = sqrt(mean_squared_error(X_test[0], X_test_pred[0][0]))
    # np.mean(RMSE)
    
    test_id = []
    x_actual = []
    y_actual = []
    x_pred  =[]
    y_pred  =[]
    rmse  =[]
    
    for i in range(len(lstm_model.x_test)):
        for j in range(len(lstm_model.x_test[i])):
            test_id1 = i
            x_actual1 = lstm_model.x_test[i][j][0]
            y_actual1 = lstm_model.x_test[i][j][1]
            x_pred1 = X_test_pred[i][0][j][0]
            y_pred1  =X_test_pred[i][0][j][1]
            rmse1  =RMSE[i]
            
            test_id.append(test_id1)
            x_actual.append(x_actual1)
            y_actual.append(y_actual1)
            x_pred.append(x_pred1)
            y_pred.append(y_pred1)
            rmse.append(rmse1)
    
    
            
    
    testing_df = pd.DataFrame({'test_id':test_id, 'x_actual':x_actual,'y_actual':y_actual,'x_pred':x_pred,'y_pred':y_pred,'rmse':rmse})
    
    
    
    traj0 = testing_df[testing_df['test_id'] == 0].sort_values('x_actual')
    traj1 = testing_df[testing_df['test_id'] == 1].sort_values('x_actual')
    traj2 = testing_df[testing_df['test_id'] == 2].sort_values('x_actual')
    traj3 = testing_df[testing_df['test_id'] == 3].sort_values('x_actual')
    traj4 = testing_df[testing_df['test_id'] == 4].sort_values('x_actual')
    traj5 = testing_df[testing_df['test_id'] == 5].sort_values('x_actual')
    traj6 = testing_df[testing_df['test_id'] == 6].sort_values('x_actual')
    traj7 = testing_df[testing_df['test_id'] == 7].sort_values('x_actual')
    traj8 = testing_df[testing_df['test_id'] == 8].sort_values('x_actual')
    traj9 = testing_df[testing_df['test_id'] == 9].sort_values('x_actual')
    
    
    traj0_p = testing_df[testing_df['test_id'] == 0].sort_values('x_pred')
    traj1_p = testing_df[testing_df['test_id'] == 1].sort_values('x_pred')
    traj2_p = testing_df[testing_df['test_id'] == 2].sort_values('x_pred')
    traj3_p = testing_df[testing_df['test_id'] == 3].sort_values('x_pred')
    traj4_p = testing_df[testing_df['test_id'] == 4].sort_values('x_pred')
    traj5_p = testing_df[testing_df['test_id'] == 5].sort_values('x_pred')
    traj6_p = testing_df[testing_df['test_id'] == 6].sort_values('x_pred')
    traj7_p = testing_df[testing_df['test_id'] == 7].sort_values('x_pred')
    traj8_p = testing_df[testing_df['test_id'] == 8].sort_values('x_pred')
    traj9_p = testing_df[testing_df['test_id'] == 9].sort_values('x_pred')
    
    
    
    
    
    tck0 = interpolate.splrep(traj0['x_actual'],traj0['y_actual'], s = 0)
    traj0['y_actual_new']=interpolate.splev(traj0['x_actual'], tck0, der=0)
    
    tck0_p = interpolate.splrep(traj0_p['x_pred'],traj0_p['y_pred'], s = 0)
    traj0_p['y_pred_new']=interpolate.splev(traj0_p['x_pred'], tck0_p, der=0)
    
    
    # sns.lmplot(data = traj0, x = 'x_actual', y = 'y_actual_new',palette=['red'], truncate=True, ci = None, order = 1000)
    # sns.lmplot(data = traj0_p, x = 'x_pred', y = 'y_pred_new', palette=['blue'])
    
    
    # sns.set_style(style="whitegrid")
    
    fig, axs = plt.subplots(2,5)
    
    sns.scatterplot(data = testing_df[testing_df['test_id'] == 0], x = 'x_actual', y = 'y_actual',ax=axs[0,0],palette=['red'])
    sns.scatterplot(data = testing_df[testing_df['test_id'] == 0], x = 'x_pred', y = 'y_pred',ax=axs[0,0], palette=['blue'])
    axs[0,0].set_xlim([6.5,9])
    axs[0,0].set_ylim([0,35])
    axs[0,0].set(xlabel="X (m)")
    axs[0,0].set(ylabel="Y (m)")
    
    sns.scatterplot(data = testing_df[testing_df['test_id'] == 1], x = 'x_actual', y = 'y_actual',ax=axs[0,1],palette=['red'])
    sns.lineplot(data = testing_df[testing_df['test_id'] == 1], x = 'x_pred', y = 'y_pred',ax=axs[0,1], palette=['blue'])
    axs[0,1].set_xlim([0,10])
    axs[0,1].set_ylim([0,35])
    axs[0,1].set(xlabel="X (m)")
    axs[0,1].set(ylabel="Y (m)")
    
    
    sns.lineplot(data = testing_df[testing_df['test_id'] == 2], x = 'x_actual', y = 'y_actual',ax=axs[0,2],palette=['red'])
    sns.lineplot(data = testing_df[testing_df['test_id'] == 2], x = 'x_pred', y = 'y_pred',ax=axs[0,2], palette=['blue'])
    axs[0,2].set_xlim([0,10])
    axs[0,2].set_ylim([0,35])
    axs[0,2].set(xlabel="X (m)")
    axs[0,2].set(ylabel="Y (m)")
    
    
    sns.lineplot(data = testing_df[testing_df['test_id'] == 3], x = 'x_actual', y = 'y_actual',ax=axs[0,3],palette=['red'])
    sns.lineplot(data = testing_df[testing_df['test_id'] == 3], x = 'x_pred', y = 'y_pred',ax=axs[0,3], palette=['blue'])
    axs[0,3].set_xlim([0,10])
    axs[0,3].set_ylim([0,35])
    axs[0,3].set(xlabel="X (m)")
    axs[0,3].set(ylabel="Y (m)")
    
    
    sns.lineplot(data = testing_df[testing_df['test_id'] == 4], x = 'x_actual', y = 'y_actual',ax=axs[0,4],palette=['red'])
    sns.lineplot(data = testing_df[testing_df['test_id'] == 4], x = 'x_pred', y = 'y_pred',ax=axs[0,4], palette=['blue'])
    axs[0,4].set_xlim([0,10])
    axs[0,4].set_ylim([0,35])
    axs[0,4].set(xlabel="X (m)")
    axs[0,4].set(ylabel="Y (m)")
    
    
    sns.lineplot(data = testing_df[testing_df['test_id'] == 5], x = 'x_actual', y = 'y_actual',ax=axs[1,0],palette=['red'])
    sns.lineplot(data = testing_df[testing_df['test_id'] == 5], x = 'x_pred', y = 'y_pred',ax=axs[1,0], palette=['blue'])
    axs[1,0].set_xlim([0,10])
    axs[1,0].set_ylim([0,35])
    axs[1,0].set(xlabel="X (m)")
    axs[1,0].set(ylabel="Y (m)")
    
    
    sns.lineplot(data = testing_df[testing_df['test_id'] == 6], x = 'x_actual', y = 'y_actual',ax=axs[1,1],palette=['red'])
    sns.lineplot(data = testing_df[testing_df['test_id'] == 6], x = 'x_pred', y = 'y_pred',ax=axs[1,1], palette=['blue'])
    axs[1,1].set_xlim([0,10])
    axs[1,1].set_ylim([0,35])
    axs[1,1].set(xlabel="X (m)")
    axs[1,1].set(ylabel="Y (m)")
    
    
    sns.lineplot(data = testing_df[testing_df['test_id'] == 7], x = 'x_actual', y = 'y_actual',ax=axs[1,2],palette=['red'])
    sns.lineplot(data = testing_df[testing_df['test_id'] == 7], x = 'x_pred', y = 'y_pred',ax=axs[1,2], palette=['blue'])
    axs[1,2].set_xlim([0,10])
    axs[1,2].set_ylim([0,35])
    axs[1,2].set(xlabel="X (m)")
    axs[1,2].set(ylabel="Y (m)")
    
    sns.lineplot(data = testing_df[testing_df['test_id'] == 8], x = 'x_actual', y = 'y_actual',ax=axs[1,3],palette=['red'])
    sns.lineplot(data = testing_df[testing_df['test_id'] == 8], x = 'x_pred', y = 'y_pred',ax=axs[1,3], palette=['blue'])
    axs[1,3].set_xlim([0,10])
    axs[1,3].set_ylim([0,35])
    axs[1,3].set(xlabel="X (m)")
    axs[1,3].set(ylabel="Y (m)")
    
    
    
    sns.lineplot(data = testing_df[testing_df['test_id'] == 9], x = 'x_actual', y = 'y_actual',ax=axs[1,4],palette=['red'])
    sns.lineplot(data = testing_df[testing_df['test_id'] == 9], x = 'x_pred', y = 'y_pred',ax=axs[1,4], palette=['blue'])
    axs[1,4].set_xlim([0,10])
    axs[1,4].set_ylim([0,35])
    axs[1,4].set(xlabel="X (m)")
    axs[1,4].set(ylabel="Y (m)")
    
    
    
    fig.savefig('visualization/example_trajs.png')
    
    
    
    
    
    # train_pre = []
    # for i in range(len(lstm_model.x_train)):
    #     a = lstm_network.predict(next(lstm_model.train_generator1))
    #     train_pre.append(a)
        
        
    # test_pre = []
    # for i in range(len(lstm_model.x_test)):
    #     a = lstm_network.predict(next(lstm_model.test_generator))
    #     test_pre.append(a)
        
        
    # for i in range(len(train_pre)):
    #     for j in range(len(train_pre[i][0])):
            
    #         train_pre[i][0][j][0] = (train_pre[i][0][j][0] * (9-6.5)) + 6.5
    #         train_pre[i][0][j][1] = (train_pre[i][0][j][1] * (37.33-1))+1
            
            
    # for i in range(len(test_pre)):
    #     for j in range(len(test_pre[i][0])):
            
    #         test_pre[i][0][j][0] = (test_pre[i][0][j][0] * (9-6.5)) + 6.5
    #         test_pre[i][0][j][1] = (test_pre[i][0][j][1] * (37.33-1))+1
            
    
    
    # for i in range(len(train_pre)):
        
            
    #         z = train_pre[i][0]
            
    #         x = z[:,0]
    #         y = z[:,1]
            
    #         filename = 'visualization/generated/train/generated_'+str(i)+'.png'
    #         plt.plot(x,y)
    #         plt.savefig(filename)
    
    
    
    # for i in range(len(test_pre)):
        
            
    #         z = test_pre[i][0]
            
    #         x = z[:,0]
    #         y = z[:,1]
            
    #         filename = 'visualization/generated/test/generated_'+str(i)+'.png'
    #         plt.plot(x,y)
    #         plt.savefig(filename)
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    
    train_z = []
    for i in range(len(lstm_model.x_train)):
        a = encoder_model.predict(next(lstm_model.train_generator1))
        train_z.append(a)
    
    train_z_array = np.array(train_z)
    train_z_array = train_z_array.reshape(1270,2)
    
    
    
    train_z_df = pd.DataFrame(train_z_array)
    
    train_z_df.columns = ['z1', 'z2']
    
    
    
    scatter_matrix(train_z_df, alpha = 0.9, figsize = (20, 20), diagonal = 'kde')
    fig.savefig('visualization/scatter_matrix.png')

    
    
    
    
    n_components = np.arange(1, 21)
    models = [GMM(n, covariance_type='full', random_state=0).fit(train_z_df) for n in n_components]
    
    plt.plot(n_components, [m.bic(train_z_df) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(train_z_df) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components');
    fig.savefig('visualization/n_components.png')

    
    
    
    gmm = GMM(n_components=4).fit(train_z_df)
    labels = gmm.predict(train_z_df)
    training_scatter_matrix = scatter_matrix(train_z_df, alpha = 0.9,c=labels, figsize = (20, 20), diagonal = 'kde')
    # fig.savefig('visualization/training_scatter_matrix.png')

    sns.set(font_scale=1)
    df1 = train_z_df
    df1['Cluster'] = labels
    c = pd.Categorical(df1['Cluster'])
    c = c.rename_categories(['Cluster 01','Cluster 02', 'Cluster 03', 'Cluster 04'])
    df1['Cluster'] = c
    
    
    p1 = sns.pairplot(df1,hue='Cluster', plot_kws = {'s':3, 'alpha':0.5})
    plt.savefig('visualization/training_scatter_matrix.png', dpi=150)
    
    
    for ax in training_scatter_matrix.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize = 20, rotation = 0, x=0.5)
        ax.set_ylabel(ax.get_ylabel(), fontsize = 20, rotation = 0, x=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize = 10, rotation = 0)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize = 10, rotation = 0)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
        x_ticklab = ax.xaxis.get_ticklabels()[0]
        y_ticklab = ax.yaxis.get_ticklabels()[0]
        x_trans = x_ticklab.get_transform()
        y_trans = y_ticklab.get_transform()
        ax.xaxis.set_label_coords(0, -0.1, transform=x_trans)
        ax.yaxis.set_label_coords(-0.2, 0, transform=y_trans)
        
    
    
    plt.savefig('visualization/training_scatter_matrix.png', dpi=50)
    
    
    data_new = gmm.sample(10000)
    data_new_df = pd.DataFrame(data_new[0])
    data_new_df.columns = ['z1','z2']
    data_new_labels = data_new[1]
    generated_scatter_matrix = scatter_matrix(data_new_df, alpha = 0.9,c=data_new_labels, figsize = (20, 20), diagonal = 'kde')
    
    for ax in generated_scatter_matrix.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize = 20, rotation = 0, x=0.5)
        ax.set_ylabel(ax.get_ylabel(), fontsize = 20, rotation = 0, x=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize = 10, rotation = 0)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize = 10, rotation = 0)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
        x_ticklab = ax.xaxis.get_ticklabels()[0]
        y_ticklab = ax.yaxis.get_ticklabels()[0]
        x_trans = x_ticklab.get_transform()
        y_trans = y_ticklab.get_transform()
        ax.xaxis.set_label_coords(0, -0.1, transform=x_trans)
        ax.yaxis.set_label_coords(-0.2, 0, transform=y_trans)
    
    
    
    sns.set(font_scale=1)
    df2 = data_new_df
    df2['Cluster'] = data_new_labels
    c = pd.Categorical(df2['Cluster'])
    c = c.rename_categories(['Cluster 01','Cluster 02', 'Cluster 03', 'Cluster 04'])
    df2['Cluster'] = c
    
    p2 = sns.pairplot(df2,hue='Cluster', plot_kws = {'s':3, 'alpha':0.5})
    plt.savefig('visualization/generated_scatter_matrix.png', dpi=50)
    
    
    import matplotlib.gridspec as gridspec
    
    fig2 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(1, 2)
    f2_ax1 = fig2.add_subplot(spec2[0,0])
    
    f2_ax2 = fig2.add_subplot(spec2[0,1])
    
    
    
    df2 =   pd.DataFrame(columns = ["x", "y", "x_m", "y_m", "ID", "z1", "z2","dx", "dy", "distance", "speed", "acceleration", "jerk", "dir_angle",  "yaw_rate", "dir_angle_abs", "yaw_rate_abs"])  
    for i in range(len(lstm_model.x_train)):
        df1 = pd.DataFrame(lstm_model.x_train[i], columns = ['x','y'])
    
        df1 = df1.assign(x_m=lambda x: df1['x']*2.5+6.5)
        df1 = df1.assign(y_m=lambda x: df1['y']*36.33+1)
        df1 = df1.assign(ID=lambda x: i)
        df1 = df1.assign(z1=lambda x: train_z_df['z1'][i])
        df1 = df1.assign(z2=lambda x: train_z_df['z2'][i])
        df1 = df1.assign(dx=lambda x: df1['x_m'] - df1['x_m'].shift(1))
        df1 = df1.assign(dy=lambda x: df1['y_m'] - df1['y_m'].shift(1))
        df1 = df1.assign(distance=lambda x: np.sqrt(df1['dx']**2+df1['dy']**2))
        df1 = df1.assign(speed=lambda x: df1['distance']*30)
        df1 = df1.assign(acceleration=lambda x: (df1['speed']-df1['speed'].shift(1))*30)
        df1 = df1.assign(jerk=lambda x: (df1['acceleration']-df1['acceleration'].shift(1))*30)
        df1 = df1.assign(dir_angle=lambda x: np.arctan((df1['dx']/df1['dy']))*180/np.pi)
        df1 = df1.assign(dir_angle=lambda x: df1['dir_angle'])
        df1 = df1.assign(yaw_rate=lambda x: (df1['dir_angle']-df1['dir_angle'].shift(1))*30)
        df1 = df1.assign(yaw_rate=lambda x: df1['yaw_rate'])
    
        df1 = df1.assign(dir_angle_abs=lambda x: (np.abs(df1['dir_angle'])))
    
        df1 = df1.assign(yaw_rate_abs=lambda x: (np.abs(df1['yaw_rate'])))
        df1 = df1.assign(label=lambda x: labels[i])
        df2 = df2.append(df1)
        
        
    df3 = df2.groupby(['ID']).mean()
    
    
    import seaborn as sns
    
    
    df_speed = pd.DataFrame({'speed' : df3['speed'], 'speed2' : df3['speed']})
    df_speed = df_speed[df_speed.speed < 15]
    df_speed = df_speed[df_speed.speed > 0]
    gmm_speed = GMM(n_components=4).fit(df_speed)
    speed_labels = gmm_speed.predict(df_speed)
    df_speed['labels'] = speed_labels
    c = pd.Categorical(df_speed['labels'])
    c = c.rename_categories(['Cluster 01','Cluster 02', 'Cluster 03', 'Cluster 04'])
    df_speed['Cluster'] = c
    
    df_acceleration = pd.DataFrame({'acceleration' : df3['acceleration'], 'acceleration2' : df3['speed']})
    df_acceleration = df_acceleration[df_acceleration.acceleration < 10]
    df_acceleration = df_acceleration[df_acceleration.acceleration > -10]
    gmm_acceleration = GMM(n_components=4).fit(df_acceleration)
    acceleration_labels = gmm_acceleration.predict(df_acceleration)
    df_acceleration['labels'] = acceleration_labels
    c = pd.Categorical(df_acceleration['labels'])
    c = c.rename_categories(['Cluster 01','Cluster 02', 'Cluster 03', 'Cluster 04'])
    df_acceleration['Cluster'] = c
    
    
    df_jerk = pd.DataFrame({'jerk' : df3['jerk'], 'jerk2' : df3['jerk']})
    df_jerk = df_jerk[df_jerk.jerk < 30]
    df_jerk = df_jerk[df_jerk.jerk > -30]
    gmm_jerk = GMM(n_components=4).fit(df_jerk)
    jerk_labels = gmm_jerk.predict(df_jerk)
    df_jerk['labels'] = jerk_labels
    c = pd.Categorical(df_jerk['labels'])
    c = c.rename_categories(['Cluster 01','Cluster 02', 'Cluster 03', 'Cluster 04'])
    df_jerk['Cluster'] = c
    
    
    df_dir_angle = pd.DataFrame({'dir_angle' : df3['dir_angle'], 'dir_angle2' : df3['dir_angle']})
    df_dir_angle = df_dir_angle[df_dir_angle.dir_angle < 10]
    df_dir_angle = df_dir_angle[df_dir_angle.dir_angle > -10]
    gmm_dir_angle = GMM(n_components=4).fit(df_dir_angle)
    dir_angle_labels = gmm_dir_angle.predict(df_dir_angle)
    df_dir_angle['labels'] = dir_angle_labels
    c = pd.Categorical(df_dir_angle['labels'])
    c = c.rename_categories(['Cluster 01','Cluster 02', 'Cluster 03', 'Cluster 04'])
    df_dir_angle['Cluster'] = c
    
    
    
    df_yaw_rate = pd.DataFrame({'yaw_rate' : df3['yaw_rate'], 'yaw_rate2' : df3['yaw_rate']})
    df_yaw_rate = df_yaw_rate[df_yaw_rate.yaw_rate < 30]
    df_yaw_rate = df_yaw_rate[df_yaw_rate.yaw_rate > -30]
    gmm_yaw_rate = GMM(n_components=4).fit(df_yaw_rate)
    yaw_rate_labels = gmm_yaw_rate.predict(df_yaw_rate)
    df_yaw_rate['labels'] = yaw_rate_labels
    c = pd.Categorical(df_yaw_rate['labels'])
    c = c.rename_categories(['Cluster 01','Cluster 02', 'Cluster 03', 'Cluster 04'])
    df_yaw_rate['Cluster'] = c
    
    
    fig, axs = plt.subplots(2,2)
    
    sns.kdeplot(data = df_speed, x="speed", hue=df_speed["Cluster"], ax = axs[0,0], bw_adjust=3, fill = True, shade=True, palette='bright')
    axs[0,0].set(xlabel="Speed (m/s)")
    axs[0,0].set_xlim([0,15])
    
    
    sns.kdeplot(data = df_acceleration, x="acceleration", hue=df_acceleration["Cluster"],  ax = axs[0,1], bw_adjust=3, fill = True, shade=True, palette='bright')
    axs[0,1].set(xlabel="Acceleration (m/s/s)")
    axs[0,1].set_xlim([-10,10])
    
    
    sns.kdeplot(data = df_jerk, x="jerk", hue=df_jerk["Cluster"], ax = axs[1,0], bw_adjust=3, fill = True, shade=True, palette='bright')
    axs[1,0].set(xlabel="Jerk (m/s/s/s)")
    axs[1,0].set_xlim([-40,40])
    
    
    sns.kdeplot(data = df_dir_angle, x="dir_angle", hue=df_dir_angle["Cluster"], ax = axs[1,1], bw_adjust=3, fill = True, shade=True, palette='bright')
    axs[1,1].set(xlabel="Direction Angle (Degrees)")
    axs[1,1].set_xlim([-10,10])
    
    
    
    
    fig.savefig('visualization/variable_distributions.png')
    
    
    
    
    
    
    df_speed = pd.DataFrame({'speed1': df3['speed'], 'speed2': df3['speed']})
    df_acceleration = pd.DataFrame({'acceleration1': df3['acceleration'], 'acceleration2': df3['acceleration']})
    df_jerk = pd.DataFrame({'jerk1': df3['jerk'], 'jerk2': df3['jerk']})
    df_dir_angle = pd.DataFrame({'dir_angle1': df3['dir_angle'], 'dir_angle2': df3['dir_angle']})
    df_yaw_rate = pd.DataFrame({'yaw_rate1': df3['yaw_rate'], 'yaw_rate2': df3['yaw_rate']})
    
    
    
    gmm = GMM(n_components=4).fit(df_speed)
    labels_speed = gmm.predict(df_speed)
    df_speed = df_speed.assign(label=lambda x: labels_speed)
    sns.displot(df_speed, x="speed1", hue="label", kind="kde")
    
    
    gmm = GMM(n_components=4).fit(df_acceleration)
    labels_acceleration = gmm.predict(df_acceleration)
    df_acceleration = df_acceleration.assign(label=lambda x: labels_acceleration)
    sns.displot(df_acceleration, x="acceleration1", hue="label", kind="kde")
    
    gmm = GMM(n_components=4).fit(df_jerk)
    labels_jerk = gmm.predict(df_jerk)
    df_jerk = df_jerk.assign(label=lambda x: labels_jerk)
    sns.displot(df_jerk, x="jerk1", hue="label", kind="kde")
    
    
    gmm = GMM(n_components=4).fit(df_dir_angle)
    labels_dir_angle = gmm.predict(df_dir_angle)
    df_dir_angle = df_dir_angle.assign(label=lambda x: labels_dir_angle)
    sns.displot(df_dir_angle, x="dir_angle1", hue="label", kind="kde")
    
    
    gmm = GMM(n_components=4).fit(df_yaw_rate)
    labels_yaw_angle = gmm.predict(df_yaw_rate)
    df_yaw_rate = df_yaw_rate.assign(label=lambda x: labels_yaw_angle)
    sns.displot(df_yaw_rate, x="yaw_rate1", hue="label", kind="kde")
    
    fig.savefig('visualization/1.png')
    
    
    
    
    
    
    
    
    
    
    
    
    train_loss_hist = []
    val_loss_hist = []
    test_loss_hist = []
    
    
    train_loss_hist2 = []
    val_loss_hist2= []
    test_loss_hist2 = []
    
    
    
    for i in range(115):
        # if i < 9:
            
        #     filename = 'save_model/run4/LPT-0'+str(i+1)+ '.h5'
        # else:
        #     filename = 'save_model/run4/LPT-'+str(i+1)+ '.h5'
        # lstm_network.load_weights(filename)
        print(i)
        train_loss = lstm_network.evaluate(lstm_model.train_generator, steps=len(lstm_model.x_train))
        val_loss = lstm_network.evaluate(lstm_model.validate_generator, steps=len(lstm_model.x_test))
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        
        
    #     with open('data_saves/train_loss_hist.pkl','wb') as f:
    #         pickle.dump(train_loss_hist, f)
            
            
    #     with open('data_saves/val_loss_hist.pkl','wb') as f:
    #         pickle.dump(val_loss_hist, f)
        
    
    
    
    
    with open('../../data_saves/train_loss_hist.pkl','rb') as f:
        train_loss_hist = pickle.load(f)
        
        
    with open('../../data_saves/val_loss_hist.pkl','rb') as f:
        val_loss_hist = pickle.load(f)
        
    # 
    epochs = list(range(0,115))
    
    
    
    # sns.set(font_scale = 2, style='whitegrid')
    fig, axs = plt.subplots()
    
    
    sns.lineplot(x = epochs, y = train_loss_hist,palette=['red'])
    sns.lineplot(x = epochs, y = val_loss_hist,palette=['blue'])
    # axs.set_xticklabels(axs.get_xticklabels(), fontsize = 14)
    # axs.set_yticklabels(axs.get_yticklabels(), fontsize = 14)
    axs.set_xlabel("Epochs")
    axs.set_ylabel("Loss")
    
    
    plt.savefig('visualization/loss.png')
    
    
if __name__ == '__main__':
    main()

