#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 08:40:42 2022

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
import unittest
from tensorflow_probability.python import bijectors as tfb
from scipy.stats.stats import pearsonr
from tensorflow_probability import distributions as tfd
from MSS.abstractmodel import MSSModel
from MSS.model1dataset import DatasetModel1SingleSensor, DatasetModel1ConcatenatedSensors
from MSS.model1 import Model1Concatenated, Model1SingleSensor, Model1SingleSensor_nocirc
from unittest import TestCase
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

class TestModel1ConcatenatedSensors(TestCase):
    """ This class assumes that DatasetModel1ConcatenatedSensors is working as
        expected"""
    def __init__(self, *args, **kwargs):
        super(TestModel1ConcatenatedSensors, self).__init__(*args, **kwargs)
        
        study_id = '06'
        sensor = '1'
        is_concatenated = '1'
        
        data_root_dir = 'test_data/'
        fname_gluc = 'glucose_MSS0'
        fname_food = 'food_MSS0'
        
        fname_gluc1 = f"{data_root_dir}{fname_gluc}{study_id}-1.csv"
        fname_gluc2 = f"{data_root_dir}{fname_gluc}{study_id}-2.csv"
        
        fname_food1 = f"{data_root_dir}{fname_food}{study_id}-1.xlsx"
        fname_food2 = f"{data_root_dir}{fname_food}{study_id}-2.xlsx"
        
        if is_concatenated == '1':
            data_gluc = DatasetModel1ConcatenatedSensors(fname_gluc1,fname_food1,fname_gluc2,fname_food2,study_id,sensor)
        elif is_concatenated == '0':
            data_gluc = DatasetModel1SingleSensor(fname_gluc1,fname_food1,study_id,sensor)
            
        if is_concatenated == '1':
            self.model1 = Model1Concatenated(data_gluc)  
        elif is_concatenated == '0':
            self.model1 = Model1SingleSensor(data_gluc)  

    def test_creation_instance(self):

        self.assertIsInstance(self.model1, Model1Concatenated)
        
    def test_sample_prior(self):
        sample_vector = self.model1.sample_prior()
        cond = []
        cond.append(sample_vector.dtype == tf.float32)
        cond.append(tf.shape(sample_vector).shape==1)
        self.assertTrue(all(cond))
        
    def test_mean_func(self):
        sample_vector = self.model1.sample_prior()
        f_meals_and_circadian, f_meals, f_circadian = self.model1.all_mean_funcs(sample_vector,self.model1.X1,self.model1.T_sub_timestamp1,self.model1.food_item_index1,1)
        cond = []
        cond.append(f_meals_and_circadian.dtype == tf.float32)
        cond.append(tf.shape(f_meals_and_circadian).shape==1)
        self.assertTrue(all(cond))
                
    def test_MAP_loss_grads(self):
        sample_vector = self.model1.sample_prior()
        loss, grads = self.model1.get_loss_and_grads_map(sample_vector)
        self.assertTrue(tf.shape(loss).shape==0)
        self.assertTrue(tf.shape(grads).shape==1)

    def test_model_fit_metrics(self):
        sample_vector = self.model1.sample_prior()
        rval_circ_and_meals, rval_meals, expl_var_circ_and_meals, expl_var_meals = self.model1.get_model_fit(sample_vector)
        cond = []
        cond.append(-1<rval_circ_and_meals<1)
        cond.append(-1<rval_meals<1)
        cond.append(expl_var_circ_and_meals<1)
        cond.append(expl_var_meals<1)
        self.assertTrue(all(cond))
        

class TestModel1SingleSensor(unittest.TestCase):
    """ This class assumes that DatasetModel1ConcatenatedSensors is working as
        expected"""
    def __init__(self, *args, **kwargs):
        super(TestModel1SingleSensor, self).__init__(*args, **kwargs)
        
        study_id = '16'
        sensor = '1'
        is_concatenated = '0'
        
        data_root_dir = 'test_data/'
        fname_gluc = 'glucose_MSS0'
        fname_food = 'food_MSS0'

        fname_gluc1 = f"{data_root_dir}{fname_gluc}{study_id}-1.csv"

        fname_food1 = f"{data_root_dir}{fname_food}{study_id}-1.xlsx"

        if is_concatenated == '1':
            data_gluc = DatasetModel1ConcatenatedSensors(fname_gluc1,fname_food1,fname_gluc2,fname_food2,study_id,sensor)
        elif is_concatenated == '0':
            data_gluc = DatasetModel1SingleSensor(fname_gluc1,fname_food1,study_id,sensor)
            
        if is_concatenated == '1':
            self.model1 = Model1Concatenated(data_gluc)  
        elif is_concatenated == '0':
            self.model1 = Model1SingleSensor(data_gluc)  

    def test_creation_instance(self):

        self.assertIsInstance(self.model1, Model1SingleSensor)
        
    def test_sample_prior(self):
        sample_vector = self.model1.sample_prior()
        cond = []
        cond.append(sample_vector.dtype == tf.float32)
        cond.append(tf.shape(sample_vector).shape==1)
        self.assertTrue(all(cond))
        
    def test_mean_func(self):
        sample_vector = self.model1.sample_prior()
        f_meals_and_circadian, f_meals, f_circadian = self.model1.all_mean_funcs(sample_vector,self.model1.X1,self.model1.T_sub_timestamp1,self.model1.food_item_index1,1)
        cond = []
        cond.append(f_meals_and_circadian.dtype == tf.float32)
        cond.append(tf.shape(f_meals_and_circadian).shape==1)
        self.assertTrue(all(cond))
                
    def test_MAP_loss_grads(self):
        sample_vector = self.model1.sample_prior()
        loss, grads = self.model1.get_loss_and_grads_map(sample_vector)
        self.assertTrue(tf.shape(loss).shape==0)
        self.assertTrue(tf.shape(grads).shape==1)

    def test_model_fit_metrics(self):
        sample_vector = self.model1.sample_prior()
        rval_circ_and_meals, rval_meals, expl_var_circ_and_meals, expl_var_meals = self.model1.get_model_fit(sample_vector)
        cond = []
        cond.append(-1<rval_circ_and_meals<1)
        cond.append(-1<rval_meals<1)
        cond.append(expl_var_circ_and_meals<1)
        cond.append(expl_var_meals<1)
        self.assertTrue(all(cond))

        
if __name__ == "__main__":
    unittest.main()
        