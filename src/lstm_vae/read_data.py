#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 00:18:21 2020

@author: hossam
"""
import csv
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split


def main():
    def read_data(file_name):
        Burrard_unconstrained = {}
        with open(file_name, mode='rU') as f:
            reader = csv.reader(f, delimiter=',')  # dialect=csv.excel_tab?
            for n, row in enumerate(reader):
                if not n:
                    # Skip header row (n = 0).
                    continue  
                Obj_ID, x, y = row
        #        x = map(float, x)
        #        y = map(float, y)
                if Obj_ID not in Burrard_unconstrained:
                    Burrard_unconstrained[Obj_ID] = list()
                Burrard_unconstrained[Obj_ID].append((x, y))
        
        
        
        
        Obj_ID_list = list(Burrard_unconstrained.keys())
            
        for j in range(len(Obj_ID_list)):
            for i in range(len(Burrard_unconstrained[Obj_ID_list[j]])):
                Burrard_unconstrained[Obj_ID_list[j]][i] = list(Burrard_unconstrained[Obj_ID_list[j]][i])
                Burrard_unconstrained[Obj_ID_list[j]][i][0] = float(Burrard_unconstrained[Obj_ID_list[j]][i][0])
                Burrard_unconstrained[Obj_ID_list[j]][i][1] = float(Burrard_unconstrained[Obj_ID_list[j]][i][1])
                
                
        Obj_ID_lengths = list()
        for i in range(len(Obj_ID_list)):
            length =len(Burrard_unconstrained[Obj_ID_list[i]])
            Obj_ID_lengths.append(length)
            
        max_length = max(Obj_ID_lengths)
        min_length = min(Obj_ID_lengths)
        
        
        Burrard_unconstrained_list = list(Burrard_unconstrained.values())
        
        Burrard_unconstrained_list2 = []
        for i in range(len(Obj_ID_lengths)):
            length_boolean = Obj_ID_lengths[i] < 10
            if length_boolean:
                Burrard_unconstrained_list2 = Burrard_unconstrained_list2
            else:
                Burrard_unconstrained_list2.append(Burrard_unconstrained_list[i])
                
                
                
        Traj_arr = Burrard_unconstrained_list2
        
        return Traj_arr
            
    
        
    def arr2df(arr):
        
        Traj_df = pd.DataFrame(columns=['ID','x','y'])
        for i in range(len(arr)):
            Traj_df_i = pd.DataFrame(columns=['ID','x','y'])
            l = len(arr[i])
            arr_reshaped = np.reshape(arr[i],(l,2)).T
            Traj_df_i['x'] = arr_reshaped[0]
            Traj_df_i['y'] = arr_reshaped[1]
            Traj_df_i['ID'] = i
            Traj_df = Traj_df.append(Traj_df_i)
    
        Traj_df.set_index('ID', inplace=True)
        return Traj_df
    
    # traj_count = pd.DataFrame(columns = ["ID", "count"])
    # for i in range(len(Traj_arr)):
    #     traj_count_i = {'ID': [i], 'count': [len(Traj_arr[i])]}
    #     traj_count_i = pd.DataFrame(traj_count_i)
    
    #     traj_count = traj_count.append(traj_count_i)
    
    # traj_count_sorted = traj_count.sort_values('count')
    
    # min(traj_count)    
    # max(traj_count)
    
    
    def TrajVar(Traj_df, Traj_arr):
        
        traj_count = pd.DataFrame(columns = ["ID", "count"])
        
        for i in range(len(Traj_arr)):
            traj_count_i = {'ID': [i], 'count': [len(Traj_arr[i])]}
            traj_count_i = pd.DataFrame(traj_count_i)
        
            traj_count = traj_count.append(traj_count_i)
    
        traj_count_sorted = traj_count.sort_values('count')
        Traj_df1 = Traj_df
        for i in range(51):
            Traj_df1 = Traj_df1.drop([traj_count_sorted.iloc[i,]['ID']])
        
        Traj_df_var =pd.DataFrame(columns = ["ID", "x", "y","dx", "dy", "distance", "speed", "acceleration", "jerk", "dir_angle",  "yaw_rate", "dir_angle_abs", "yaw_rate_abs", "Time"])
        for i in Traj_df1.index.unique():
            Traj_df_i = Traj_df1.loc[i]
            Traj_df_i["ID"] = i
            Traj_df_i = Traj_df_i.assign(dx=lambda x: Traj_df_i['x'] - Traj_df_i['x'].shift(1))
            Traj_df_i = Traj_df_i.assign(dy=lambda x: Traj_df_i['y'] - Traj_df_i['y'].shift(1))
            Traj_df_i = Traj_df_i.assign(distance=lambda x: np.sqrt(Traj_df_i['dx']**2+Traj_df_i['dy']**2))
            Traj_df_i = Traj_df_i.dropna()
            Traj_df_i = Traj_df_i.assign(speed=lambda x: signal.savgol_filter(Traj_df_i['distance']*30,13,2))
            Traj_df_i = Traj_df_i.dropna()
            Traj_df_i = Traj_df_i.assign(speed_diff=lambda x: (Traj_df_i['speed'].shift(1)-Traj_df_i['speed'])*30)
            Traj_df_i = Traj_df_i.dropna()
            Traj_df_i = Traj_df_i.assign(acceleration=lambda x: signal.savgol_filter(Traj_df_i['speed_diff'],5,2))
            Traj_df_i = Traj_df_i.dropna()
            Traj_df_i = Traj_df_i.assign(acc_diff=lambda x: (Traj_df_i['acceleration'].shift(1)-Traj_df_i['acceleration'])*30)
            Traj_df_i = Traj_df_i.dropna()
            Traj_df_i = Traj_df_i.assign(jerk=lambda x: signal.savgol_filter(Traj_df_i['acc_diff'],5,2))
            Traj_df_i = Traj_df_i.dropna()
            Traj_df_i = Traj_df_i.assign(dir_angle=lambda x: np.arctan(Traj_df_i['dx']/Traj_df_i['dy']*180/np.pi))
            Traj_df_i = Traj_df_i.assign(dir_angle=lambda x: signal.savgol_filter(Traj_df_i['dir_angle'],5,2))
            Traj_df_i = Traj_df_i.dropna()
            Traj_df_i = Traj_df_i.assign(dir_angle_diff=lambda x: (Traj_df_i['dir_angle'].shift(1)-Traj_df_i['dir_angle'])*30)
            Traj_df_i = Traj_df_i.dropna()
            Traj_df_i = Traj_df_i.assign(yaw_rate=lambda x: signal.savgol_filter(Traj_df_i['dir_angle_diff'],5,2))
            Traj_df_i = Traj_df_i.assign(dir_angle_abs=lambda x: (np.abs(Traj_df_i['dir_angle'])))
            Traj_df_i = Traj_df_i.assign(yaw_rate_abs=lambda x: (np.abs(Traj_df_i['yaw_rate'])))
            Traj_df_i = Traj_df_i.assign(Time=lambda x: range(len(Traj_df_i)))
    
            Traj_df_var = Traj_df_var.append(Traj_df_i)
            Traj_df_var =  Traj_df_var.dropna()
            
        return Traj_df_var
    
    
    def TrajVar_no_smooth(Traj_df, Traj_arr):
        
        traj_count = pd.DataFrame(columns = ["ID", "count"])
        
        for i in range(len(Traj_arr)):
            traj_count_i = {'ID': [i], 'count': [len(Traj_arr[i])]}
            traj_count_i = pd.DataFrame(traj_count_i)
        
            traj_count = traj_count.append(traj_count_i)
    
        traj_count_sorted = traj_count.sort_values('count')
        Traj_df1 = Traj_df
        for i in range(51):
            Traj_df1 = Traj_df1.drop([traj_count_sorted.iloc[i,]['ID']])
        
        Traj_df_var =pd.DataFrame(columns = ["ID", "x", "y","dx", "dy", "distance", "speed", "acceleration", "jerk", "dir_angle",  "yaw_rate", "dir_angle_abs", "yaw_rate_abs", "Time"])
        for i in Traj_df1.index.unique():
            Traj_df_i = Traj_df1.loc[i]
            Traj_df_i["ID"] = i
            Traj_df_i = Traj_df_i.assign(dx=lambda x: Traj_df_i['x'] - Traj_df_i['x'].shift(1))
            Traj_df_i = Traj_df_i.assign(dy=lambda x: Traj_df_i['y'] - Traj_df_i['y'].shift(1))
            Traj_df_i = Traj_df_i.assign(distance=lambda x: np.sqrt(Traj_df_i['dx']**2+Traj_df_i['dy']**2))
            Traj_df_i = Traj_df_i.dropna()
            Traj_df_i = Traj_df_i.assign(speed=lambda x: Traj_df_i['distance']*30)
            Traj_df_i = Traj_df_i.dropna()
            Traj_df_i = Traj_df_i.assign(acceleration=lambda x: (Traj_df_i['speed'].shift(1)-Traj_df_i['speed'])*30)
            Traj_df_i = Traj_df_i.dropna()
            Traj_df_i = Traj_df_i.assign(jerk=lambda x: (Traj_df_i['acceleration'].shift(1)-Traj_df_i['acceleration'])*30)
            Traj_df_i = Traj_df_i.dropna()
            Traj_df_i = Traj_df_i.assign(dir_angle=lambda x: np.arctan(Traj_df_i['dx']/Traj_df_i['dy']*180/np.pi))
            Traj_df_i = Traj_df_i.dropna()
            Traj_df_i = Traj_df_i.assign(yaw_rate=lambda x: (Traj_df_i['dir_angle'].shift(1)-Traj_df_i['dir_angle'])*30)
            Traj_df_i = Traj_df_i.dropna()
            Traj_df_i = Traj_df_i.assign(dir_angle_abs=lambda x: (np.abs(Traj_df_i['dir_angle'])))
            Traj_df_i = Traj_df_i.assign(yaw_rate_abs=lambda x: (np.abs(Traj_df_i['yaw_rate'])))
            Traj_df_i = Traj_df_i.assign(Time=lambda x: range(len(Traj_df_i)))
    
            Traj_df_var = Traj_df_var.append(Traj_df_i)
            Traj_df_var =  Traj_df_var.dropna()
            
        return Traj_df_var
    
    
    
    
    
    def Traj_df_var_filter(Traj_df_var):
    
        Traj_df_var_grouped = Traj_df_var.groupby('ID').agg({'speed': 'mean', 'acceleration': 'mean', 'jerk': 'mean', 'yaw_rate': 'mean'})
        Traj_df_var_grouped_sorted_speed = Traj_df_var_grouped.sort_values(by=['speed'], ascending=False)
        
        Traj_df_var_grouped_sorted_speed = Traj_df_var_grouped_sorted_speed.iloc[20:]
        
        
        Traj_df_var_grouped_sorted_acceleration = Traj_df_var_grouped_sorted_speed.sort_values(by=['acceleration'], ascending=False)
        
        Traj_df_var_grouped_sorted_acceleration = Traj_df_var_grouped_sorted_acceleration.iloc[10:]
        Traj_df_var_grouped_sorted_acceleration = Traj_df_var_grouped_sorted_acceleration.iloc[:-10]
        
        
        
        Traj_df_var_grouped_sorted_jerk = Traj_df_var_grouped_sorted_acceleration.sort_values(by=['jerk'], ascending=False)
        Traj_df_var_grouped_sorted_jerk = Traj_df_var_grouped_sorted_jerk.iloc[10:]
        Traj_df_var_grouped_sorted_jerk = Traj_df_var_grouped_sorted_jerk.iloc[:-10]
        
        
        Traj_df_var_grouped_sorted_yaw_rate = Traj_df_var_grouped_sorted_jerk.sort_values(by=['yaw_rate'], ascending=False)
        
        Traj_df_var_filtered = Traj_df_var.loc[Traj_df_var.index.isin(Traj_df_var_grouped_sorted_yaw_rate.index)]
        
        Traj_df_var_filtered2 = Traj_df_var_filtered[Traj_df_var_filtered['speed'] < 10]
        Traj_df_var_filtered2 = Traj_df_var_filtered2[Traj_df_var_filtered2['speed'] > 0]
        Traj_df_var_filtered2 = Traj_df_var_filtered2[Traj_df_var_filtered2['acceleration'] < 20]
        Traj_df_var_filtered2 = Traj_df_var_filtered2[Traj_df_var_filtered2['acceleration'] > -20]
        Traj_df_var_filtered2 = Traj_df_var_filtered2[Traj_df_var_filtered2['jerk'] < 200]
        Traj_df_var_filtered2 = Traj_df_var_filtered2[Traj_df_var_filtered2['jerk'] > -200]
        
        return(Traj_df_var_filtered2)
    
    def Traj_arr_filtered(Traj_df_var_filtered):
        Traj_arr_filtered = []
        for i in np.unique(Traj_df_var_filtered.index):
            Traj_df_var_filtered_i = Traj_df_var_filtered.loc[i]
            Traj_df_var_filtered_i = Traj_df_var_filtered_i[["x", "y"]]
            Traj_df_var_filtered_i = Traj_df_var_filtered_i.to_numpy()
            Traj_arr_filtered.append(Traj_df_var_filtered_i)
        return Traj_arr_filtered
    
    
    
    def data_split(Traj_arr1, random_state = 333, test_size = 0.33):
    
        data_index = list(range(len(Traj_arr1)))
        X_train_index, X_test_index = train_test_split(data_index,test_size=test_size, random_state=random_state)
        
        X_train = []
        X_test = []
        
        for i in X_train_index:
            X_train.append(Traj_arr1[i])
            
        for i in X_test_index:
            X_test.append(Traj_arr1[i])
    
    
            
        return X_train, X_test
    
    
    
    Traj_arr = read_data('data/Burrard_unconstrained.csv')
    Traj_df = arr2df(Traj_arr)
    Traj_df_var = TrajVar(Traj_df, Traj_arr)
    Traj_df_var_no_smooth = TrajVar_no_smooth(Traj_df, Traj_arr)
    Traj_df_var_filtered = Traj_df_var_filter(Traj_df_var)
    Traj_arr_filtered1 = Traj_arr_filtered(Traj_df_var_filtered)
    X_train_init, X_test = data_split(Traj_arr_filtered1, random_state = 333, test_size = 0.2)
    X_train, X_validate = data_split(X_train_init, random_state = 333, test_size = 0.33)

    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
        
            X_train[i][j][0] = (X_train[i][j][0]-6.5) /(9-6.5)
            X_train[i][j][1] = (X_train[i][j][1]-1)/(37.33-1)
        
        
    for i in range(len(X_validate)):
        for j in range(len(X_validate[i])):
            
            X_validate[i][j][0] = (X_validate[i][j][0]-6.5) /(9-6.5)
            X_validate[i][j][1] = (X_validate[i][j][1]-1)/(37.33-1)
            
    np.savez('data/processed_data.npz', X_train=X_train, X_test=X_test, X_validate=X_validate)



if __name__ == "__main__":
    main()

