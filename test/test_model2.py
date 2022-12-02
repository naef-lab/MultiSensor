#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:28:57 2022

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
from MSS.model2dataset import DatasetModel2Actiheart
from MSS.model2 import Model2Actiheart
from unittest import TestCase
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

class TestModel2Actiheart(TestCase):
    """ This class assumes that DatasetModel2Actiheart is working as
        expected"""
    def __init__(self, *args, **kwargs):
        super(TestModel2Actiheart, self).__init__(*args, **kwargs)
        
        study_id = '27'     
        acti_root_dir = 'test_data/'
        fname_acti = 'actiheart_MSS0'
        fname_acti1 = f"{acti_root_dir}{fname_acti}{study_id}.csv"
        data = DatasetModel2Actiheart(fname_acti1)
        self.model2 = Model2Actiheart(data)  

    def test_creation_instance(self):

        self.assertIsInstance(self.model2, Model2Actiheart)
        
    def test_sample_prior(self):
        sample_vector = self.model2.sample_prior()
        cond = []
        cond.append(sample_vector.dtype == tf.float32)
        cond.append(tf.shape(sample_vector).shape==1)
        self.assertTrue(all(cond))
        
    def test_mean_func(self):
        sample_vector = self.model2.sample_prior()
        mean_act, mean_bpm, mean_hrv, _, _, _, _ = self.model2.get_transition_probs(sample_vector,self.model2.X)
        meanf = tf.stack([mean_act,mean_bpm,mean_hrv], 0)
        cond = []
        cond.append(meanf.dtype == tf.float32)
        cond.append(tf.shape(meanf)[0]==3)
        cond.append(tf.shape(meanf)[1]>0)
        self.assertTrue(all(cond))
                
    def test_MAP_loss_grads(self):
        sample_vector = self.model2.sample_prior()
        loss, grads = self.model2.get_loss_and_grads_map(sample_vector)
        self.assertTrue(tf.shape(loss).shape==0)
        self.assertTrue(tf.shape(grads).shape==1)

    def test_model_fit_metrics(self):
        sample_vector = self.model2.sample_prior()
        expl_var_circ, expl_var_circ_and_act = self.model2.get_model_fit_hr(sample_vector)
        cond = []
        cond.append(expl_var_circ<1)
        cond.append(expl_var_circ_and_act<1)
        expl_var_circ, expl_var_circ_and_act, expl_var_circ_act_hrv = self.model2.get_model_fit_hrv(sample_vector)
        cond.append(expl_var_circ<1)
        cond.append(expl_var_circ_and_act<1)
        cond.append(expl_var_circ_act_hrv<1)
        self.assertTrue(all(cond))
          
if __name__ == "__main__":
    unittest.main()