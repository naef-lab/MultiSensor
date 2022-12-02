#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 08:48:34 2022

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
from MSS.model2dataset import DatasetModel2Actiheart
from MSS.model2 import Model2Actiheart
from unittest import TestCase
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

class TestDatasetModel2Actiheart(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDatasetModel2Actiheart, self).__init__(*args, **kwargs)

        study_id = '27'     
        acti_root_dir = 'test_data/'
        fname_acti = 'actiheart_MSS0'
        fname_acti1 = f"{acti_root_dir}{fname_acti}{study_id}.csv"
        self.data = DatasetModel2Actiheart(fname_acti1)
        
    def test_creation_instance(self):

        self.assertIsInstance(self.data, DatasetModel2Actiheart)
        
    def test_dtype(self):
        cond = []
        cond.append(self.data.X.dtype == tf.float32)
        cond.append(self.data.Y.dtype == tf.float32)
        cond.append(self.data.mask.dtype == 'bool')
        self.assertTrue(all(cond))
        
    def test_data_shape(self):
        cond = []
        cond.append(tf.shape(self.data.Y)[1]==3)
        cond.append(tf.shape(self.data.Y)[0]>0)
        cond.append(tf.shape(self.data.X).shape==1)
        self.assertTrue(all(cond))  
        
if __name__ == "__main__":
    unittest.main()
        