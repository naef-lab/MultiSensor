#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:44:32 2022

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
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


class DatasetModel1ConcatenatedSensors:
    """
    Container that holds both glucose and meal data for an individual with
    two glucose sensors. 
    This dataset class is used as an input into Model1, and this class 
    also contains several plotting functions for visualisation. 

    :param fname_gluc1: the glucose filename from the first sensor
    :type fname_gluc1: str
    :param fname_food1: the food filename associated with the first sensor
    :type fname_food1: str
    :param fname_gluc2: the glucose filename from the second sensor
    :type fname_gluc2: str
    :param fname_food2: the food filename associated with the second sensor
    :type fname_food2: str
    """
    
    def __init__(
        self,
        fname_gluc1: str,
        fname_food1: str,
        fname_gluc2: str,
        fname_food2: str,
        study_id: str,
        sensor: str,
        ) -> None:  
        """Initialize Dataset object for Model1 with two glucose sensors."""
        
        self.fname_gluc1 = fname_gluc1
        self.fname_food1 = fname_food1
        self.fname_gluc2 = fname_gluc2
        self.fname_food2 = fname_food2
        self.study_id = study_id
        self.sensor = sensor
        self._load_data()
            
    def _load_data(self):
        """
        Loads in food and glucose data using provided file names
        """
              
        gluc_db1 = pd.read_csv(self.fname_gluc1,index_col=False)
        gluc_db2 = pd.read_csv(self.fname_gluc2,index_col=False)
        
        if self.fname_food1.endswith('.xlsx'):
            food_db1 = pd.read_excel(self.fname_food1)
        elif self.fname_food1.endswith('.csv'):
            food_db1 = pd.read_csv(self.fname_food1)
        else:
            raise DataInputError("Invalid file format. The file should be either .xlsx or .csv.")

        if self.fname_food2.endswith('.xlsx'):
            food_db2 = pd.read_excel(self.fname_food2)
        elif self.fname_food2.endswith('.csv'):
            food_db2 = pd.read_csv(self.fname_food2)
        else:
            raise DataInputError("Invalid file format. The file should be either .xlsx or .csv.")
        
        cond1 = (gluc_db1.shape[0] ==0)
        cond2 = (gluc_db2.shape[0] ==0)
        cond3 = (food_db1.shape[0] ==0)
        cond4 = (food_db2.shape[0] ==0)
        if cond1|cond2|cond3|cond4:
            raise DataInputError(
                "Food and glucose input files "
                + "must have at least one entry"
            )
        
        gluc_var_list = ['abs_time_hours','Detrended','hours','days']
        food_var_list = ['food_item_index','abs_time_hours']
        
        cond1 = all(value in list(gluc_db1.keys()) for value in gluc_var_list)
        cond2 = all(value in list(gluc_db2.keys()) for value in gluc_var_list)
        if not cond1&cond2:
            raise DataInputError(
                "Glucose input file must contain the following columns: "
                + "'abs_time_hours','Detrended','hours','days'"
            )    
        
        cond1 = all(value in list(food_db1.keys()) for value in food_var_list)
        cond2 = all(value in list(food_db2.keys()) for value in food_var_list)
        if not cond1&cond2:
            raise DataInputError(
                "Food input file must contain the following columns: "
                + "'food_item_index','abs_time_hours'"
            )  
            
        self.X1 = tf.convert_to_tensor(gluc_db1['abs_time_hours'].values, 
                                       dtype=tf.float32)
        self.Y1 = tf.convert_to_tensor(gluc_db1['Detrended'].values, 
                                       dtype=tf.float32)[:,None]
        self.hours1 = gluc_db1['hours'].values
        self.days1 = gluc_db1['days'].values
    
        self.food_item_index1 = tf.constant(food_db1['food_item_index'].values)[:,tf.newaxis]
        
        self.timestamps1 = food_db1['abs_time_hours'].values
        T1 = np.repeat(self.X1.numpy().reshape(1,-1),len(self.timestamps1),
                       axis = 0)  
        self.T_sub_timestamp1 = tf.convert_to_tensor((T1 - self.timestamps1.reshape(-1,1)),
                                                     dtype=tf.float32)  
        
        self.X2 = tf.convert_to_tensor(gluc_db2['abs_time_hours'].values, 
                                       dtype=tf.float32)
        self.Y2 = tf.convert_to_tensor(gluc_db2['Detrended'].values, 
                                       dtype=tf.float32)[:,None]
        self.hours2 = gluc_db2['hours'].values
        self.days2 = gluc_db2['days'].values
    
        self.food_item_index2 = tf.constant(food_db2['food_item_index'].values)[:,tf.newaxis]
        
        self.timestamps2 = food_db2['abs_time_hours'].values
        T2 = np.repeat(self.X2.numpy().reshape(1,-1),len(self.timestamps2),
                       axis = 0)  
        self.T_sub_timestamp2 = tf.convert_to_tensor((T2 - self.timestamps2.reshape(-1,1)),
                                                     dtype=tf.float32)  
    
        self.N_meals1 = np.max(food_db1['food_item_index'])+1
        self.N_meals2 = np.max(food_db2['food_item_index'])+1
        self.N_global = 10

    def _find_nearest(self,array, value):
        """
        Function that finds the element in array that is closest to value.
        Used to help align the time axis of two different glucose sensors
        for visualisation purposes.
    
        :param array: the reference array 
        :param value: the target value
        :returns: the element of the array that is closest to value
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
        
    def plot_glucose_circadian(self,x_label=True,y_label=True):
        """ Shows all glucose data projected onto the 24-hour time axis.
        Different days are represented with different colours, and the mean
        is shown in black. When the data comes from two different sensors,
        the circadian mean is found by first aligning the time grid of 
        the second sensor with the first.
        
        :param x_label: if True, the xlabel 'Time (hours)' will be shown
        :param y_label: if True, the ylabel 'Glucose (mmol/L)' will be shown
        """
        
        days = np.concatenate((self.days1,self.days2))
        hours = np.concatenate((self.hours1,self.hours2))
        y_tot = np.concatenate((self.Y1,self.Y2))
        
        for i in np.arange(0,max(days)):
            filt = days==i
            x = hours[filt]
            y = y_tot[filt]
            cmap = plt.get_cmap("tab20")
            c = cmap(i%max(days))
            plt.plot(x,y,alpha=0.3,color=c)

        x_hours1 = self.hours1
        x_hours2 = self.hours2
        x_hours2_realigned = np.zeros_like(x_hours2)
        
        for i, x_curr in enumerate(x_hours2):
            x_hours2_realigned[i] = self._find_nearest(x_hours1, x_curr)   
        
        x_hours = np.concatenate((x_hours1,x_hours2_realigned))
        hours_unique = np.unique(x_hours)
        y_mean = np.zeros_like(hours_unique)
        
        for i, hour_unique in enumerate(hours_unique):
            filt = x_hours==hour_unique
            y = y_tot[filt]
            y_mean[i] = np.mean(y)
        
        plt.plot(hours_unique,y_mean,'k--')
        
        plt.xlim((0,24))
        plt.xticks(np.arange(0, 25, step=4))
        plt.ylim([2,10])
        plt.yticks(np.arange(2, 11, step=2))
        
        if x_label:
            plt.xlabel('Time (hours)')
        if y_label:
            plt.ylabel('Glucose (mmol/L)')
        
    def plot_glucose_overview(self,x_label=True,y_label=True):
        """ Shows a time series of glucose data over all recorded days
        
        :param x_label: if True, the xlabel 'Time (days)' will be shown
        :param y_label: if True, the ylabel 'Glucose (mmol/L)' will be shown
        """
        
        plt.plot(self.X1/24,self.Y1,color=config.COLOR_GLUC,linewidth=1)
        plt.plot(self.X2/24,self.Y2,color=config.COLOR_GLUC,linewidth=1)
        plt.ylim([2,10])
        plt.yticks(np.arange(2, 11, step=2))  
        if x_label:
            plt.xlabel('Time (days)')
        if y_label:
            plt.ylabel('Glucose (mmol/L)')

    def plot_gluc_single_day(self,day,x_label=True,y_label=True):
        """ Shows a time series of glucose and food timestamp data for a 
        s single, user-specified day
        
        :param day: the day to show the single-day example
        :type day: int
        :param x_label: if True, the xlabel 'Time (days)' will be shown
        :param y_label: if True, the ylabel 'Glucose (mmol/L)' will be shown
        """
        
        assert type(day) == int, "day must be an integer"
        
        days = np.concatenate((self.days1,self.days2))
        hours = np.concatenate((self.hours1,self.hours2))
        y_tot = np.concatenate((self.Y1,self.Y2))
        timestamps = np.concatenate((self.timestamps1,self.timestamps2))
        
        filt = days==day
        x = hours[filt]
        y = y_tot[filt]
        
        plt.xlim((0,24))
        plt.xticks(np.arange(0, 25, step=4))
        plt.ylim([2,10])
        plt.yticks(np.arange(2, 11, step=2))
        
        plt.plot(x,y,color= config.COLOR_GLUC,alpha=0.8)
        
        for timestamp in timestamps:      
            plt.plot([timestamp,timestamp],[2,10],'k:',alpha=0.3)   
        
        if x_label:
            plt.xlabel('Time (hours)')
        if y_label:
            plt.ylabel('Glucose (mmol/L)')

class DatasetModel1SingleSensor:
    """
    Container that holds both glucose and meal data for an individual. 
    This dataset class is used as an input into Model1, and this class 
    also contains several plotting functions for visualisation. 

    :param fname_gluc1: the glucose filename from the first sensor
    :type fname_gluc1: str
    :param fname_food1: the food filename associated with the first sensor
    :type fname_food1: str
    """
    
    def __init__(
        self,
        fname_gluc1: str,
        fname_food1: str,
        study_id: str,
        sensor: str,
        ) -> None:  
        """Initialize Dataset object for Model1 with two glucose sensors."""
        
        self.fname_gluc1 = fname_gluc1
        self.fname_food1 = fname_food1
        self.study_id = study_id
        self.sensor = sensor
        self._load_data()
            
    def _load_data(self):
        """
        Loads in food and glucose data using provided file names
        """
              
        gluc_db1 = pd.read_csv(self.fname_gluc1,index_col=False)
        
        if self.fname_food1.endswith('.xlsx'):
            food_db1 = pd.read_excel(self.fname_food1)
        elif self.fname_food1.endswith('.csv'):
            food_db1 = pd.read_csv(self.fname_food1)
        else:
            raise DataInputError("Invalid file format. The file should be either .xlsx or .csv.")

        cond1 = (gluc_db1.shape[0] ==0)
        cond2 = (food_db1.shape[0] ==0)
        if cond1|cond2:
            raise DataInputError(
                "Food and glucose input files "
                + "must have at least one entry"
            )
        
        gluc_var_list = ['abs_time_hours','Detrended','hours','days']
        food_var_list = ['food_item_index','abs_time_hours']
        
        cond1 = all(value in list(gluc_db1.keys()) for value in gluc_var_list)
        if not cond1:
            raise DataInputError(
                "Glucose input file must contain the following columns: "
                + "'abs_time_hours','Detrended','hours','days'"
            )    
        
        cond1 = all(value in list(food_db1.keys()) for value in food_var_list)
        if not cond1:
            raise DataInputError(
                "Food input file must contain the following columns: "
                + "'food_item_index','abs_time_hours'"
            )  
            
        self.X1 = tf.convert_to_tensor(gluc_db1['abs_time_hours'].values, 
                                       dtype=tf.float32)
        self.Y1 = tf.convert_to_tensor(gluc_db1['Detrended'].values, 
                                       dtype=tf.float32)[:,None]
        self.hours1 = gluc_db1['hours'].values
        self.days1 = gluc_db1['days'].values
    
        self.food_item_index1 = tf.constant(food_db1['food_item_index'].values)[:,tf.newaxis]
        self.food_hours = food_db1['hours'].values
        self.food_days = food_db1['days'].values
        
        self.timestamps1 = food_db1['abs_time_hours'].values
        T1 = np.repeat(self.X1.numpy().reshape(1,-1),len(self.timestamps1),axis = 0)  
        self.T_sub_timestamp1 = tf.convert_to_tensor((T1 - self.timestamps1.reshape(-1,1)),
                                                     dtype=tf.float32)  
        
        self.N_meals1 = np.max(food_db1['food_item_index'])+1
        self.N_global = 10
        
    def plot_glucose_circadian(self,x_label=True,y_label=True):
        """ Shows all glucose data projected onto the 24-hour time axis.
        Different days are represented with different colours, and the mean
        is shown in black. When the data comes from two different sensors,
        the circadian mean is found by first aligning the time grid of 
        the second sensor with the first.
        
        :param x_label: if True, the xlabel 'Time (hours)' will be shown
        :param y_label: if True, the ylabel 'Glucose (mmol/L)' will be shown
        """
        
        days = self.days1
        hours = self.hours1
        y_tot = self.Y1
        
        for i in np.arange(0,max(days)):
            filt = days==i
            x = hours[filt]
            y = y_tot[filt]
            cmap = plt.get_cmap("tab20")
            c = cmap(i%max(days))
            plt.plot(x,y,alpha=0.3,color=c)

        hours_unique = np.unique(hours)
        y_mean = np.zeros_like(hours_unique)
        
        for i, hour_unique in enumerate(hours_unique):
            filt = hours==hour_unique
            y = y_tot[filt]
            y_mean[i] = np.mean(y)
        
        plt.plot(hours_unique,y_mean,'k--')
        
        plt.xlim((0,24))
        plt.xticks(np.arange(0, 25, step=4))
        plt.ylim([2,10])
        plt.yticks(np.arange(2, 11, step=2))
        
        if x_label:
            plt.xlabel('Time (hours)')
        if y_label:
            plt.ylabel('Glucose (mmol/L)')
        
    def plot_glucose_overview(self,x_label=True,y_label=True):
        """ Shows a time series of glucose data over all recorded days
        
        :param x_label: if True, the xlabel 'Time (days)' will be shown
        :param y_label: if True, the ylabel 'Glucose (mmol/L)' will be shown
        """
        
        plt.plot(self.X1/24,self.Y1,color=config.COLOR_GLUC,linewidth=1)
        plt.ylim([2,10])
        plt.yticks(np.arange(2, 11, step=2))  
        if x_label:
            plt.xlabel('Time (days)')
        if y_label:
            plt.ylabel('Glucose (mmol/L)')

    def plot_gluc_single_day(self,day,x_label=True,y_label=True):
        """ Shows a time series of glucose and food timestamp data for a 
        s single, user-specified day
        
        :param day: the day to show the single-day example
        :type day: int
        :param x_label: if True, the xlabel 'Time (days)' will be shown
        :param y_label: if True, the ylabel 'Glucose (mmol/L)' will be shown
        """
        
        assert type(day) == int, "day must be an integer"
        
        days = self.days1
        hours = self.hours1
        y_tot = self.Y1
        food_days = self.food_days
        
        filt = days==day
        x = hours[filt]
        y = y_tot[filt]
        filt_food = food_days==day
        timestamps = self.food_hours[filt_food]
        
        plt.xlim((0,24))
        plt.xticks(np.arange(0, 25, step=4))
        plt.ylim([2,11])
        plt.yticks(np.arange(2, 11, step=2))
        
        plt.plot(x,y,color= config.COLOR_GLUC,alpha=0.8)
        
        for timestamp in timestamps:      
            plt.plot([timestamp,timestamp],[2,11],'k:',alpha=0.3)   
        
        if x_label:
            plt.xlabel('Time (hours)')
        if y_label:
            plt.ylabel('Glucose (mmol/L)')   
    
class DataInputError(RuntimeError):
    pass
