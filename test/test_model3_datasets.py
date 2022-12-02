#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:55:34 2022

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
from MSS.model3dataset import DatasetModel3Integrated
from MSS.model3 import Model3Integrated
from unittest import TestCase
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

class TestDatasetModel3Integrated(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDatasetModel3Integrated, self).__init__(*args, **kwargs)

        fname_all = 'test_data/glucose_actiheart_integrated_MSS16-1.csv'
        fname_food = 'test_data/food_integrated_MSS16-1.xlsx'

        self.data = DatasetModel3Integrated(fname_all,fname_food,sensor='1')
        
    def test_creation_instance(self):

        self.assertIsInstance(self.data, DatasetModel3Integrated)
        
    def test_dtype(self):
        cond = []
        cond.append(self.data.X.dtype == tf.float32)
        cond.append(self.data.Y.dtype == tf.float32)
        cond.append(self.data.mask.dtype == 'bool')
        self.assertTrue(all(cond))
        
    def test_data_shape(self):
        cond = []
        cond.append(tf.shape(self.data.Y)[1]==4)
        cond.append(tf.shape(self.data.Y)[0]>0)
        cond.append(tf.shape(self.data.X).shape==1)
        self.assertTrue(all(cond))  
        
if __name__ == "__main__":
    unittest.main()