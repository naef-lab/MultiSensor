#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 03:30:42 2022

@author: phillips
"""

import os
import dill as pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow_probability as tfp
import MSS.config as config
import matplotlib.ticker as ticker

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

class DatasetModel3Integrated:
    """ Container that holds the meal, glucose and 
    Actiheart data for an individual, which
    includes physical activity, heart rate and heart rate variability.
    This dataset class is used as an input into Model3.

    :param fname_all: the glucose filename from the first sensor
    :type fname_all: str
    """  
    def __init__(
            self,
            fname_all: str,
            fname_food: str,
            sensor: str,
        ) -> None:
        
            # Setting random seeds
            self.fname_all = fname_all
            self.fname_food = fname_food
            self.sensor = sensor
            self._load_data()
            
    def _load_data(self):
        """
        Loads in food, glucose, activity, HR and HRV data using provided 
            file names
        """
        data_all = pd.read_csv(self.fname_all,index_col=False)
        
        mask = data_all['mask'].values
        start_indx = np.argmin(mask)
        reversed_mask = mask[::-1]
        end_indx = np.min((-np.argmin(reversed_mask),-1))#-np.argmin(reversed_mask)

        x1 = data_all['abs_time_hours'].values.reshape(-1, 1)
        y0 = (data_all['Activity'].values**(1/2)).reshape(-1, 1)
        y1 = data_all['BPM'].values.reshape(-1, 1)
        y2 = 1/data_all['RMSSD'].values.reshape(-1, 1)

        Y_acti = np.hstack([y0,y1,y2]).T
        Y_filted = Y_acti[:,~mask]
        normalisation_constant = np.std(Y_filted,1)
        Y_acti = Y_acti/normalisation_constant[:,None]
        y3 = data_all['Detrended'].values.reshape(-1, 1)
        Y = np.vstack([Y_acti,y3.T])

        x1 = x1[start_indx:end_indx]
        Y = Y[:,start_indx:end_indx]
        mask = mask[start_indx:end_indx]
        
        food_db = pd.read_excel(self.fname_food)
        
        food_item_index1 = food_db['food_item_index'].values
        food_item_index = tf.constant(food_item_index1)[:,tf.newaxis]
        
        N_meals = np.max(food_db['food_item_index'])+1
        timestamps = food_db['abs_time_hours'].values
        T1 = np.repeat(x1.reshape(1,-1),len(timestamps),axis = 0)  
        T_sub_timestamp = tf.convert_to_tensor((T1 - timestamps.reshape(-1,1)),
                                               dtype=tf.float32)  

        X = np.squeeze(x1.astype(np.float32))
        Y = Y.astype(np.float32)
        
        self.X = tf.convert_to_tensor(X, dtype=tf.float32)
        self.Y = tf.transpose(tf.convert_to_tensor(Y, dtype=tf.float32))
        
        self.mask = mask
        self.normalisation_constant = normalisation_constant
        self.timestamps = timestamps
        self.T_sub_timestamp = T_sub_timestamp
        self.food_item_index = food_item_index
        self.N_meals = N_meals

