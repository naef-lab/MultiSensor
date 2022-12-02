#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 08:23:45 2022

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
import unittest
from tensorflow_probability.python import bijectors as tfb
from scipy.stats.stats import pearsonr
from tensorflow_probability import distributions as tfd
from MSS.abstractmodel import MSSModel
from MSS.model1dataset import DatasetModel1SingleSensor, DatasetModel1ConcatenatedSensors
from MSS.model1 import Model1SingleSensor
from unittest import TestCase
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

class TestDatasetModel1ConcatenatedSensors(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDatasetModel1ConcatenatedSensors, self).__init__(*args, **kwargs)
        
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
            self.data = DatasetModel1ConcatenatedSensors(fname_gluc1,fname_food1,fname_gluc2,fname_food2,study_id,sensor)
        elif is_concatenated == '0':
            self.data = DatasetModel1SingleSensor(fname_gluc1,fname_food1,study_id,sensor)
            
    def test_creation_instance(self):

        self.assertIsInstance(self.data, DatasetModel1ConcatenatedSensors)
        
    def test_dtype(self):
        cond = []
        cond.append(self.data.X1.dtype == tf.float32)
        cond.append(self.data.Y1.dtype == tf.float32)
        cond.append(self.data.T_sub_timestamp1.dtype == tf.float32)
        cond.append(self.data.X2.dtype == tf.float32)
        cond.append(self.data.Y2.dtype == tf.float32)
        cond.append(self.data.T_sub_timestamp2.dtype == tf.float32)
        self.assertTrue(all(cond))
        
    def test_data_shape(self):
        cond = []
        cond.append(tf.shape(self.data.Y1)[1]==1)
        cond.append(tf.shape(self.data.Y1)[0]>0)
        cond.append(tf.shape(self.data.X1).shape==1)
        cond.append(tf.shape(self.data.Y2)[1]==1)
        cond.append(tf.shape(self.data.Y2)[0]>0)
        cond.append(tf.shape(self.data.X2).shape==1)
        self.assertTrue(all(cond))
       
class TestDatasetModel1SingleSensor(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDatasetModel1SingleSensor, self).__init__(*args, **kwargs)
        
        study_id = '16'
        sensor = '1'
        is_concatenated = '0'
        
        data_root_dir = 'test_data/'
        fname_gluc = 'glucose_MSS0'
        fname_food = 'food_MSS0'

        fname_gluc1 = f"{data_root_dir}{fname_gluc}{study_id}-1.csv"
        fname_food1 = f"{data_root_dir}{fname_food}{study_id}-1.xlsx"

        if is_concatenated == '1':
            self.data = DatasetModel1ConcatenatedSensors(fname_gluc1,fname_food1,fname_gluc2,fname_food2,study_id,sensor)
        elif is_concatenated == '0':
            self.data = DatasetModel1SingleSensor(fname_gluc1,fname_food1,study_id,sensor)
            
    def test_creation_instance(self):
        self.assertIsInstance(self.data, DatasetModel1SingleSensor)
        
    def test_dtype(self):
        cond = []
        cond.append(self.data.X1.dtype == tf.float32)
        cond.append(self.data.Y1.dtype == tf.float32)
        cond.append(self.data.T_sub_timestamp1.dtype == tf.float32)
        self.assertTrue(all(cond))
        
    def test_data_shape(self):
        cond = []
        cond.append(tf.shape(self.data.Y1)[1]==1)
        cond.append(tf.shape(self.data.Y1)[0]>0)
        cond.append(tf.shape(self.data.X1).shape==1)
        self.assertTrue(all(cond))        
        
if __name__ == "__main__":
    unittest.main()
        
        