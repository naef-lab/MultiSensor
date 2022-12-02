#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 20:07:23 2022

@author: phillips
"""
import os
import dill as pickle
import pandas as pd
import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow_probability as tfp
import MSS.config as config
import matplotlib.ticker as ticker

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

class DatasetModel2Actiheart:
    """
    Container that holds the Actiheart data for an individual, which
    includes physical activity, heart rate and heart rate variability.
    This dataset class is used as an input into Model2, and this class 
    also contains several plotting functions for visualisation. 

    :param fname_acti: the glucose filename from the first sensor
    :type fname_acti: str
    """    
    def __init__(
            self,
            fname_acti
        ) -> None:
        
            # Setting random seeds
            self.fname_acti = fname_acti
            self._load_data()
            
    def _load_data(self):
        """
        Loads in Actiheart data using provided file names
        """
        acti_df = pd.read_csv(self.fname_acti,index_col=False)
        filt_acti = (acti_df['Comments'] !='Invalid data') & (acti_df['BPM']!=0) & (acti_df['Quality'] >=0.8) 
        acti_df['filt'] = filt_acti 
        self.acti_df = acti_df

        x1 = acti_df['abs_time_hours'].values.reshape(-1, 1)
        y0 = (acti_df['Activity'].values**(1/2)).reshape(-1, 1)
        y1 = acti_df['BPM'].values.reshape(-1, 1)
        y2 = 1/acti_df['RMSSD'].values.reshape(-1, 1)

        mask = ~filt_acti.values
        start_indx = np.argmin(mask)
        reversed_mask = mask[::-1]
        end_indx = np.min((-np.argmin(reversed_mask),-1))

        Y = np.hstack([y0,y1,y2]).T
        
        x1 = x1[start_indx:end_indx]
        Y = Y[:,start_indx:end_indx]
        self.mask = mask[start_indx:end_indx]

        Y_filted = Y[:,~self.mask]
        self.normalisation_constant = np.std(Y_filted,1)
        
        Y = Y/self.normalisation_constant[:,None]
        
        X = np.squeeze(x1.astype(np.float32))
        Y = Y.astype(np.float32)
        
        self.X = tf.convert_to_tensor(X, dtype=tf.float32)
        self.Y = tf.transpose(tf.convert_to_tensor(Y, dtype=tf.float32))
        
    def plot_activity_single_day(self,day,x_label=True,y_label=True):
        """ Shows a time series of physical activity data for a 
        single, user-specified day
        
        :param day: the day to show the single-day example
        :type day: int
        :param x_label: if True, the xlabel 'Time (days)' will be shown
        :param y_label: if True, the ylabel 'Glucose (mmol/L)' will be shown
        """
        
        assert type(day) == int, "day must be an integer"
        
        acti_curr = self.acti_df[self.acti_df['days']==day]
        filt_curr = acti_curr['filt'].values
        
        var = 'Activity'
        
        x = acti_curr['hours'].values
        y = acti_curr[var].values
        
        plt.scatter(x[filt_curr],y[filt_curr],color=config.COLOR_ACTI,alpha=0.5,s=1)
        
        y_nan = y.astype(np.float64)
        y_nan[~filt_curr] = np.nan
        
        plt.plot(x,y_nan,color=config.COLOR_ACTI,alpha=0.3)
        
        plt.ylim([-100,5000])
        plt.yticks(np.arange(0, 6000, step=1000))
        
        plt.xlim([0,24])
        plt.xticks(np.arange(0, 25, step=4)) 
        plt.gca().axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x/1000)) + 'k'))
        
        if x_label:
            plt.xlabel('Time (hours)')
        if y_label:
            plt.ylabel('Activity counts')  
            
    def plot_hr_single_day(self,day,x_label=True,y_label=True):
        """ Shows a time series of heart rate data for a 
        single, user-specified day
        
        :param day: the day to show the single-day example
        :type day: int
        :param x_label: if True, the xlabel 'Time (days)' will be shown
        :param y_label: if True, the ylabel 'Glucose (mmol/L)' will be shown
        """
        
        assert type(day) == int, "day must be an integer"
        
        acti_curr = self.acti_df[self.acti_df['days']==day]
        filt_curr = acti_curr['filt'].values
        
        var = 'BPM'
        
        x = acti_curr['hours'].values
        y = acti_curr[var].values
        
        plt.scatter(x[filt_curr],y[filt_curr],color=config.COLOR_HR,alpha=0.5,s=1)
        
        y_nan = y.astype(np.float64)
        y_nan[~filt_curr] = np.nan
        
        plt.plot(x,y_nan,color=config.COLOR_HR,alpha=0.3)
        
        plt.ylim([50,180])
        plt.yticks(np.arange(50, 190, step=20))
        
        plt.xlim([0,24])
        plt.xticks(np.arange(0, 25, step=4)) 

        if x_label:
            plt.xlabel('Time (hours)')
        if y_label:
            plt.ylabel('HR (bpm)')  
            
    def plot_heart_single_day(self,day,x_label=True,y_label1=True,y_label2=True,legend=True):
        """ Shows a time series of physical activity data for a 
        single, user-specified day
        
        :param day: the day to show the single-day example
        :type day: int
        :param x_label: if True, the xlabel 'Time (days)' will be shown
        :param y_label: if True, the ylabel 'Glucose (mmol/L)' will be shown
        """
        
        assert type(day) == int, "day must be an integer"
        
        acti_curr = self.acti_df[self.acti_df['days']==day]
        filt_curr = acti_curr['filt'].values
        
        var = 'BPM'
        
        x = acti_curr['hours'].values
        y = acti_curr[var].values
        
        ax1=plt.gca()
        ax1.scatter(x[filt_curr],y[filt_curr],color=config.COLOR_HR,alpha=0.5,s=1)
        y_nan = y.astype(np.float64)
        y_nan[~filt_curr] = np.nan
        lns1 = ax1.plot(x,y_nan,color=config.COLOR_HR,alpha=0.3,label='HR')
        
        ax1.set_ylim([0,180])
        ax1.set_yticks(np.arange(0, 190, step=25))
        
        ax1.set_xlim([0,24])
        ax1.set_xticks(np.arange(0, 25, step=4)) 
        
        var = 'RMSSD'
        y = 1/acti_curr[var].values
        ax2 = ax1.twinx()
        ax2.scatter(x[filt_curr],y[filt_curr],color=config.COLOR_HRV,alpha=0.5,s=1)
        y_nan = y.astype(np.float64)
        y_nan[~filt_curr] = np.nan
        lns2 = ax2.plot(x,y_nan,color=config.COLOR_HRV,alpha=0.3,label='HRV')

        ax2.set_ylim([0,0.6])
        ax2.set_yticks(np.arange(0, 0.31, step=0.1))
        
        if legend:
            lns = lns1+lns2
            labs = [l.get_label() for l in lns]
            plt.legend(lns, labs, loc=0)
        if x_label:
            ax1.set_xlabel('Time (hours)')
        if y_label1:
            ax1.set_ylabel('HR (bpm)')  
        if y_label2:
            ax2.set_ylabel('HRV (ms$^{-1}$)',position=(1,0.15)) 
        elif y_label2==False:
            ax2.set_yticklabels([])