#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import dill as pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow_probability as tfp
import MSS.config as config
import gpflow
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
f64 = gpflow.utilities.to_default_float

def detrend_glucose(x,y,lengthscale=2.):
    x1_raw = x.reshape(-1, 1)
    y1_raw = y.reshape(-1, 1)

    k = gpflow.kernels.SquaredExponential()
    m = gpflow.models.GPR(data=(x1_raw, y1_raw-np.mean(y1_raw)), kernel=k, mean_function=None)
    m.kernel.lengthscales.assign(2.)
    gpflow.set_trainable(m.kernel.lengthscales, flag = False)
    opt = gpflow.optimizers.Scipy()    
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

    xx = x1_raw
    mean, var = m.predict_f(xx)

    trend = mean.numpy()+np.mean(y1_raw)
    detrended = y1_raw-mean.numpy()
    
    return trend, detrended
    