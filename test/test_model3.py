#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:06:26 2022

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
from MSS.model3dataset import DatasetModel3Integrated
from MSS.model3 import Model3Integrated
from unittest import TestCase
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

class TestModel3Integrated(TestCase):
    """ This class assumes that DatasetModel3Integrated is working as
        expected"""
    def __init__(self, *args, **kwargs):
        super(TestModel3Integrated, self).__init__(*args, **kwargs)
        
        fname_all = 'test_data/glucose_actiheart_integrated_MSS16-1.csv'
        fname_food = 'test_data/food_integrated_MSS16-1.xlsx'

        data = DatasetModel3Integrated(fname_all,fname_food,sensor='1')
        
        with open('test_data/model216.pickle', 'rb') as handle:
            model2 = pickle.load(handle)  
            
        samples_list = model2.samples_list
        posterior = tf.concat(samples_list,0) 
        posterior_mean_actiheart = tf.reduce_mean(posterior,0)

        self.model3 = Model3Integrated(data,posterior_mean_actiheart)

    def test_creation_instance(self):

        self.assertIsInstance(self.model3, Model3Integrated)
        
    def test_sample_prior(self):
        sample_vector = self.model3.sample_prior()
        cond = []
        cond.append(sample_vector.dtype == tf.float32)
        cond.append(tf.shape(sample_vector).shape==1)
        self.assertTrue(all(cond))
        
    def test_mean_func(self):
        sample_vector = self.model3.sample_prior()
        mean_act, mean_bpm, mean_hrv, mean_gluc, _, _, _, _ = self.model3.get_transition_probs(sample_vector,self.model3.X,self.model3.T_sub_timestamp,self.model3.food_item_index,C_51_only=False,C_52_only=False,C_53_only=False)
        meanf = tf.stack([mean_act,mean_bpm,mean_hrv,mean_gluc], 0)
        cond = []
        cond.append(meanf.dtype == tf.float32)
        cond.append(tf.shape(meanf)[0]==4)
        cond.append(tf.shape(meanf)[1]>0)
        self.assertTrue(all(cond))
                
    def test_MAP_loss_grads(self):
        sample_vector = self.model3.sample_prior()
        loss, grads = self.model3.get_loss_and_grads_map(sample_vector)
        self.assertTrue(tf.shape(loss).shape==0)
        self.assertTrue(tf.shape(grads).shape==1)

    def test_model_fit_metrics(self):
        sample_vector = self.model3.sample_prior()
        expl_var_circ, expl_var_circ_and_meals, expl_var_activity_hr_hrv, expl_var_diff = self.model3.get_model_fit(sample_vector)
        cond = []
        cond.append(expl_var_circ<1)
        cond.append(expl_var_circ_and_meals<1)
        cond.append(expl_var_activity_hr_hrv<1)
        cond.append(expl_var_diff<1)
        self.assertTrue(all(cond))
         
if __name__ == "__main__":
    unittest.main()