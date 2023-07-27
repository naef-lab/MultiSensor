#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:19:38 2022

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
from tensorflow_probability.python import bijectors as tfb
from scipy.stats.stats import pearsonr
from tensorflow_probability import distributions as tfd
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from .abstractmodel import MSSModel

class Model1Concatenated(MSSModel):
    """ Class that implements Model1, which models glucose CGM and
        meal timestamp data. This class is for the case when there are two
        CGM sensors. The model is based on a 2-D system of
        stochastic differential equations, which is implemented using the
        TensorFlow Probability distribution LinearGaussianStateSpaceModel. 
        
        :param DatasetModel1Concatenated: a DatasetModel1Concatenated class
            that contains all data for the CGM sensors and meal timestamps
    """
    def __init__(
            self,
            DatasetModel1Concatenated,
        ) -> None:
        
        self.__dict__ = DatasetModel1Concatenated.__dict__
        
        self.epsilon = 1e-6
        self.w = tf.constant(2*np.pi/24)
        self.prior = tfd.JointDistributionSequentialAutoBatched([
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[5.]),
                                         bijector=tfb.Log(), name='A_0'),
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),
                                         bijector=tfb.Log(), name='A_1'),
             tfd.VonMises(loc=[0.0], concentration=[0.0], name='phi_1'),
             tfp.distributions.Normal(loc = [0.0],scale = [0.5], name='A_11'),
             tfp.distributions.Normal(loc = [0.0],scale = [0.5], name='A_12'),
             tfp.distributions.Normal(loc = [0.0],scale = [0.5], name='A_21'),   
             tfp.distributions.Normal(loc = [0.0],scale = [0.5], name='A_22'),      
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.5]),
                                         bijector=tfb.Log(), name='tau'),
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.5]),
                                         bijector=tfb.Log(), name='B_11'),
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.5]),
                                         bijector=tfb.Log(), name='B_22'),
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=5*tf.ones(self.N_meals1,dtype=tf.float32)),
                                         bijector=tfb.Log(), name='meal_heights1'),
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=5*tf.ones(self.N_meals2,dtype=tf.float32)),
                                         bijector=tfb.Log(), name='meal_heights2'),
             ]) 

        super().__init__()
                
    def transform_params(self,params):
        """ Transforms an unconstrained TensorFlow array into named variables
            with constraints applied
            
        :param params: an unconstrained TensorFlow array of parameter values
        :return: named parameter variables with constraints applied
        """
        A_0 = tf.math.exp(params[0,tf.newaxis])
        A_1 = tf.math.exp(params[1,tf.newaxis])
        phi_1 = params[2,tf.newaxis]
        A_11 = tf.math.exp(params[3,tf.newaxis])
        A_21 = tf.math.exp(params[5,tf.newaxis])
        A_12 = tf.math.exp(params[4,tf.newaxis])
        A_22 = tf.math.exp(params[6,tf.newaxis])
        tau = tf.math.exp(params[7,tf.newaxis])
        B_11 = 0.
        B_22 = tf.math.exp(params[8,tf.newaxis])
        sigma = tf.math.exp(params[9,tf.newaxis]) 
        meal_heights1 = tf.math.exp(params[self.N_global:(self.N_global+self.N_meals1),tf.newaxis])
        meal_heights2 = tf.math.exp(params[(self.N_global+self.N_meals1):(self.N_global+self.N_meals1+self.N_meals2),tf.newaxis])
        
        return A_0, A_1, phi_1, A_11, A_12, A_21, A_22, tau, B_11, B_22, sigma, meal_heights1, meal_heights2
        
    def sample_prior(self):
        """ Draws a sample from the model prior, returned as a TensorFlow array
        
        :returns: a real Tensor 
        :rtype: tf.float32
        """    
        
        prior_sample = self.prior.sample()
        sample_vector = tf.concat(prior_sample,0)
        return sample_vector  
    
    def unpack_vector(self,params):
        """ Converts from an unconstrained TensorFlow array to a list, 
            which is compatible with the TensorFlow Probability 
            distribution JointDistributionSequentialAutoBatched
            
        :param params: an unconstrained TensorFlow array of parameter values
        :return: list of parameter variables
        """
        params_unpacked = [params[0,tf.newaxis],
                           params[1,tf.newaxis],
                           params[2,tf.newaxis],
                           params[3,tf.newaxis],
                           params[4,tf.newaxis],
                           params[5,tf.newaxis],
                           params[6,tf.newaxis],
                           params[7,tf.newaxis],
                           params[8,tf.newaxis],
                           params[9,tf.newaxis],
                           params[self.N_global:(self.N_global+self.N_meals1)],
                           params[(self.N_global+self.N_meals1):(self.N_global+self.N_meals1+self.N_meals2)]]
        return params_unpacked
    
    def mean_meals(self,A_0,A_1,phi_1,s,gamma,A_21,tau,meal_heights,X,T_sub_timestamp,food_item_index):
        """ Using the parameters of the model and information on the timestamps
            of recorded meals, this function returns the meal function r(t)
            
        :param A_0: the baseline glucose value
        :param A_1: the amplitude of the oscillation
        :param phi_1: the peak time of the oscillation
        :param s: parameter related to the eigenvalues of the Model1 A matrix
        :param gamma: parameter related to the eigenvalues of the Model1 A matrix
        :param A_21: of the Model1 A matrix
        :param tau: the time delay between meal ingestion and the start of
            the response function
        :param meal_heights: the heights of the meal response function. There
            is one height associated with each meal
        :param X: the time vector of input glucose CGM data
        :param T_sub_timestamp: an array that subtracts the recorded meal times
            from the time vector X
        :param food_item_index: a list of integers that control if meals share
            the same height response function. This is the case when meals
            have the same free text annotation
        :return: the meal response function    
        """
        normalisation_factor = tf.cond(tf.math.greater_equal(gamma, 0),
                     lambda : (A_21*tf.exp(tf.sqrt(gamma)*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)+s*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)))/(2*tf.sqrt(gamma)) - (A_21*tf.exp(-tf.sqrt(gamma)*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)+s*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)))/(2*tf.sqrt(gamma)),
                     lambda : (A_21*tf.exp(s*tf.atan(-tf.sqrt(-gamma)/s)/tf.sqrt(-gamma))*tf.sin(tf.sqrt(-gamma)*tf.atan(-tf.sqrt(-gamma)/s)/tf.sqrt(-gamma)))/tf.sqrt(-gamma))
    
        meal_heights = meal_heights/normalisation_factor
            
        T_heaviside = tf.experimental.numpy.heaviside(T_sub_timestamp-tau,1)
        t = T_heaviside*(T_sub_timestamp-tau)
        
        perturbation_fn = tf.cond(tf.math.greater_equal(gamma, 0),
                             lambda : (A_21*tf.exp(tf.sqrt(gamma)*t+s*t))/(2*tf.sqrt(gamma)) - (A_21*tf.exp(-tf.sqrt(gamma)*t+s*t))/(2*tf.sqrt(gamma)),
                             lambda : (A_21*tf.exp(s*t)*tf.sin(tf.sqrt(-gamma)*t))/tf.sqrt(-gamma))
        
        heights_unpacked = tf.gather_nd(meal_heights,food_item_index)
                
        return A_0 + tf.reduce_sum(heights_unpacked*perturbation_fn,0)
    
    def mean_circadian(self,A_0,A_1,phi_1,X):
        """ The circadian rhythm in glucose values
        
        :param A_0: the baseline glucose value
        :param A_1: the amplitude of the oscillation
        :param phi_1: the peak time of the oscillation
        :return: the circadian function
        """
        return A_0 + A_1*(1+tf.cos(self.w*X-phi_1))/2 
    
    def mean_meals_and_circadian(self,A_0,A_1,phi_1,s,gamma,A_21,tau,
                                 meal_heights,X,T_sub_timestamp,food_item_index):
        """ Using the parameters of the model and information on the timestamps
            of recorded meals, this function returns the sum of the
            meal function and the circadian baseline oscillation
            
        :param A_0: the baseline glucose value
        :param A_1: the amplitude of the oscillation
        :param phi_1: the peak time of the oscillation
        :param s: parameter related to the eigenvalues of the Model1 A matrix
        :param gamma: parameter related to the eigenvalues of the Model1 A matrix
        :param A_21: of the Model1 A matrix
        :param tau: the time delay between meal ingestion and the start of
            the response function
        :param meal_heights: the heights of the meal response function. There
            is one height associated with each meal
        :param X: the time vector of input glucose CGM data
        :param T_sub_timestamp: a matrix that subtracts the recorded meal times
            from the time vector X
        :param food_item_index: a list of integers that control if meals share
            the same height response function. This is the case when meals
            have the same free text annotation
        :return: the sum of the meal response function and the circadian baseline    
        """    
        f_meals = self.mean_meals(A_0,A_1,phi_1,s,gamma,A_21,tau,
                                  meal_heights,X,T_sub_timestamp,food_item_index)
        f_circadian = self.mean_circadian(A_0,A_1,phi_1,X)
        
        return f_meals+f_circadian-A_0
    
    def all_mean_funcs(self,params,X,T_sub_timestamp,food_item_index,sensor):
        """ Using the parameters of the model and information on the timestamps
            of recorded meals, this function returns the sum of the
            meal function and the circadian baseline oscillation
            
        :param params: an unconstrained TensorFlow array of parameter values
        :param X: the time vector of input glucose CGM data
        :param T_sub_timestamp: a matrix that subtracts the recorded meal times
            from the time vector X
        :param food_item_index: a list of integers that control if meals share
            the same height response function. This is the case when meals
            have the same free text annotation
        :return: the meal response function, the circadian baseline and the sum 
            of the meal response function and the circadian baseline    
        """  
        
        A_0, A_1, phi_1, A_11, A_12, A_21, A_22, tau, B_11, B_22, sigma, meal_heights1, meal_heights2 = self.transform_params(params)
        meal_heights = tf.cond(tf.math.equal(sensor, 1),
                         lambda : meal_heights1,
                         lambda : meal_heights2)
    
        s = -(A_11+A_22)/2
        gamma = (A_11/2 + A_22/2)**2 - A_11*A_22 - A_12*A_21
        
        f_meals_and_circadian = self.mean_meals_and_circadian(A_0,A_1,phi_1,s,gamma,
                                                              A_21,tau,meal_heights,
                                                              X,T_sub_timestamp,
                                                              food_item_index)
        f_meals = self.mean_meals(A_0,A_1,phi_1,s,gamma,A_21,tau,
                                  meal_heights,X,T_sub_timestamp,food_item_index)
        f_circadian = self.mean_circadian(A_0,A_1,phi_1,X)
        
        return f_meals_and_circadian, f_meals, f_circadian
      
    def model1_ssm(self,params,X,Y,T_sub_timestamp,food_item_index,sensor):
        """ Creates Model1 as a TensorFlow Probability 
        LinearGaussianStateSpaceModel distribution. The log liklihood
        of this model is subsequently used for parameter inference
            
        :param params: an unconstrained TensorFlow array of parameter values
        :param X: the time vector of input glucose CGM data (in absolute hours)
        :param T_sub_timestamp: an array that subtracts the recorded meal times
            from the time vector X
        :param food_item_index: a list of integers that control if meals share
            the same height response function. This is the case when meals
            have the same free text annotation
        :return: Model1 as a LinearGaussianStateSpaceModel object   
        """ 
        
        Delta_t = X[1]-X[0]
        A_0, A_1, phi_1, A_11, A_12, A_21, A_22, tau, B_11, B_22, sigma, meal_heights1, meal_heights2 = self.transform_params(params)
        
        if sensor == 1:
            meal_heights = meal_heights1
        elif sensor == 2:
            meal_heights = meal_heights2
                
        s = -(A_11+A_22)/2
        gamma = (A_11/2 + A_22/2)**2 - A_11*A_22 - A_12*A_21
        
        meanf = self.mean_meals_and_circadian(A_0,A_1,phi_1,s,gamma,A_21,tau,
                                              meal_heights,X,T_sub_timestamp,
                                              food_item_index)
    
        def f_expm_M(s,gamma):
          return tf.cond(tf.math.greater_equal(gamma, 0),
                         lambda : tf.concat([tf.stack([-tf.exp(s*Delta_t)*((A_11*tf.sinh(tf.sqrt(gamma)*Delta_t))/tf.sqrt(gamma) - tf.cosh(tf.sqrt(gamma)*Delta_t) + (s*tf.sinh(tf.sqrt(gamma)*Delta_t))/tf.sqrt(gamma)),-(A_12*tf.exp(s*Delta_t)*tf.sinh(tf.sqrt(gamma)*Delta_t))/tf.sqrt(gamma)], 1),tf.stack([(A_21*tf.exp(s*Delta_t)*tf.sinh(tf.sqrt(gamma)*Delta_t))/tf.sqrt(gamma), -tf.exp(s*Delta_t)*((A_22*tf.sinh(tf.sqrt(gamma)*Delta_t))/tf.sqrt(gamma) - tf.cosh(tf.sqrt(gamma)*Delta_t) + (s*tf.sinh(tf.sqrt(gamma)*Delta_t))/tf.sqrt(gamma))], 1)],0),
                         lambda : tf.concat([tf.stack([-tf.exp(s*Delta_t)*((A_11*tf.sin(tf.sqrt(-gamma)*Delta_t))/tf.sqrt(-gamma) - tf.cos(tf.sqrt(-gamma)*Delta_t) + (s*tf.sin(tf.sqrt(-gamma)*Delta_t))/tf.sqrt(-gamma)),-(A_12*tf.exp(s*Delta_t)*tf.sin(tf.sqrt(-gamma)*Delta_t))/tf.sqrt(-gamma)], 1),tf.stack([(A_21*tf.exp(s*Delta_t)*tf.sin(tf.sqrt(-gamma)*Delta_t))/tf.sqrt(-gamma), -tf.exp(s*Delta_t)*((A_22*tf.sin(tf.sqrt(-gamma)*Delta_t))/tf.sqrt(-gamma) - tf.cos(tf.sqrt(-gamma)*Delta_t) + (s*tf.sin(tf.sqrt(-gamma)*Delta_t))/tf.sqrt(-gamma))], 1)],0))
    
        expm_M = f_expm_M(s,gamma)
        P = tf.concat([tf.stack([(B_11*A_11**2*A_22 + A_21*B_11*A_11*A_12 + B_11*A_11*A_22**2 + B_22*A_12**2*A_22)/((A_11*A_22 + A_12*A_21)*(A_11 + A_22)),(A_11*A_22*(A_21*B_11 - A_12*B_22))/((A_11*A_22 + A_12*A_21)*(A_11 + A_22))], 1),tf.stack([(A_11*A_22*(A_21*B_11 - A_12*B_22))/((A_11*A_22 + A_12*A_21)*(A_11 + A_22)), (B_22*A_11**2*A_22 + B_11*A_11*A_21**2 + B_22*A_11*A_22**2 + A_12*B_22*A_21*A_22)/((A_11*A_22 + A_12*A_21)*(A_11 + A_22))], 1)],0)
        initial_loc = tf.stack([0.,0.],0)
        initial_scale_diag = tf.concat([(B_11*A_11**2*A_22 + A_21*B_11*A_11*A_12 + B_11*A_11*A_22**2 + B_22*A_12**2*A_22)/((A_11*A_22 + A_12*A_21)*(A_11 + A_22)),(B_22*A_11**2*A_22 + B_11*A_11*A_21**2 + B_22*A_11*A_22**2 + A_12*B_22*A_21*A_22)/((A_11*A_22 + A_12*A_21)*(A_11 + A_22))],0)        
        transition_cov = P - tf.linalg.matmul(expm_M,tf.linalg.matmul(P,tf.transpose(expm_M)))+ tf.linalg.diag([self.epsilon,self.epsilon])  
          
        transition_matrix = expm_M
        transition_noise=tfd.MultivariateNormalTriL(
            scale_tril=tf.linalg.cholesky(transition_cov))
        
        def observation_noise(t):
            loc = meanf[t]
            return tfd.MultivariateNormalDiag(
                    loc=loc,
                    scale_diag=sigma**2 * tf.ones([1]))        
        
        model = tfd.LinearGaussianStateSpaceModel(
              num_timesteps=len(X),
              transition_matrix=transition_matrix,
              transition_noise=transition_noise,
              observation_matrix=tf.stack([[0., 1.]], 0),
               observation_noise=observation_noise,
              initial_state_prior=tfd.MultivariateNormalDiag(
               loc =  initial_loc,     
               scale_diag=initial_scale_diag))
        
        return model    
    
    def MAP_model1_ssm(self,params):
        """ The negative of the unnormalized posterior distribution. The 
        negative is taken as the optimisation algorithm will minimise the
        objective function i.e. it will find the maximum of the 
        unnormalized posterior distribution (the MAP)
        
        :param params: an unconstrained TensorFlow array of parameter values
        :return: the negative of the unnormalized posterior distribution
        """
        
        return -self.unnormalized_posterior(params) 
    
    def unnormalized_posterior(self,params):
        """ The unnormalized posterior distribution. This is used for MCMC
        sampling
        
        :param params: an unconstrained TensorFlow array of parameter values
        :return: the unnormalized posterior distribution
        """
        unpacked_vector = self.unpack_vector(params)
        loss_prior = self.prior.log_prob(unpacked_vector)
        
        model1 = self.model1_ssm(params,self.X1,self.Y1,self.T_sub_timestamp1,
                                 self.food_item_index1,1)
        loss_time_series1 = model1.log_prob(self.Y1)
        
        model2 = self.model1_ssm(params,self.X2,self.Y2,self.T_sub_timestamp2,
                                 self.food_item_index2,2)
        loss_time_series2 = model2.log_prob(self.Y2)
           
        return loss_time_series1 + loss_time_series2 + loss_prior   
    
    def lik_model1_ssm(self,params):
        """ The log probability density of Model1
        
        :param params: an unconstrained TensorFlow array of parameter values
        :return: the log probability density of Model1
        """
        model1 = self.model1_ssm(params,self.X1,self.Y1,self.T_sub_timestamp1,
                                 self.food_item_index1,1)
        loss_time_series1 = model1.log_prob(self.Y1)
        
        model2 = self.model1_ssm(params,self.X2,self.Y2,self.T_sub_timestamp2,
                                 self.food_item_index2,2)
        loss_time_series2 = model2.log_prob(self.Y2)
           
        return loss_time_series1 + loss_time_series2 
    
    @tf.function(jit_compile=True)
    def get_loss_and_grads_map(self,params):
        """The loss and gradients of the Model1 MAP function, which are 
        needed by the tfp.optimizer.bfgs_minimize algorithm used for
        finding the MAP solution
        
        :param params: an unconstrained TensorFlow array of parameter values
        :return: the loss and gradients of the MAP function
        """
        with tf.GradientTape() as tape:
            tape.watch(params)
            loss = self.MAP_model1_ssm(params)
        grads = tape.gradient(loss,params)
        return loss, grads
    
    def check_meals_glucose_fit(self,params,legend=True):
        """ For a given set of parameters, this function will compare the
        mean of Model1 (using the sum of the circadian and meal functions)
        with the data
        
        """
        plt.subplot(2,1,1)

        f_meals_and_circadian, _, _ = self.all_mean_funcs(params,
                                                          self.X1,
                                                          self.T_sub_timestamp1,
                                                          self.food_item_index1,1)
        
        
        plt.plot(self.X1,f_meals_and_circadian,color=config.COLOR_PRED1,
                                 linewidth=1,label='Meal+circ. model')
        plt.plot(self.X1,self.Y1,color=config.COLOR_GLUC,
                                 linewidth=1,label='CGM data')
        
        for timestamp in self.timestamps1:
            plt.plot([timestamp,timestamp],[2,10],'k:',alpha=0.1,linewidth=1)
            
        rval, pval = pearsonr(f_meals_and_circadian,self.Y1[:,0])
        plt.title('R = %(correlation)1.2f' % {"correlation": rval})
        plt.xlabel('Time (h)')
        plt.ylabel('Glucose (mmol/L)')
        
        plt.subplot(2,1,2)

        f_meals_and_circadian, _, _ = self.all_mean_funcs(params,
                                                          self.X2,
                                                          self.T_sub_timestamp2,
                                                          self.food_item_index2,2)

        plt.plot(self.X2,f_meals_and_circadian,color=config.COLOR_PRED1,
                                 linewidth=1,label='Meal+circ. model')
        plt.plot(self.X2,self.Y2,color=config.COLOR_GLUC,
                                 linewidth=1,label='CGM data')
        
        for timestamp in self.timestamps2:
            plt.plot([timestamp,timestamp],[2,10],'k:',alpha=0.1,linewidth=1)
        if legend:
            plt.legend()
        rval, pval = pearsonr(f_meals_and_circadian,self.Y2[:,0])
        plt.title('R = %(correlation)1.2f' % {"correlation": rval})
        plt.xlabel('Time (h)')
        plt.ylabel('Glucose (mmol/L)')
        
        plt.tight_layout()

    def transform_posterior_params(self,samples_list):
        """ Creates a dictionary of posterior parameter samples for named
            parameters
            
        :param samples_list: a list of samples from the posterior parameter
            distribution
        :return: a dictionary that contains samples of the the posterior
            distribution for named parameters
        """
        posterior = tf.concat(samples_list,0) 
        
        A_0 = tf.math.exp(posterior[:,0])
        A_1 = tf.math.exp(posterior[:,1])
        phi_1 = posterior[:,2]
        A_11 = tf.math.exp(posterior[:,3])
        A_21 = tf.math.exp(posterior[:,5])
        A_12 = tf.math.exp(posterior[:,4])
        A_22 = tf.math.exp(posterior[:,6])
        tau = tf.math.exp(posterior[:,7])
        B_22 = tf.math.exp(posterior[:,8])
        sigma = tf.math.exp(posterior[:,9])
        meal_heights = tf.math.exp(posterior[:,self.N_global:])
        
        s = -(A_11+A_22)/2
        gamma = (A_11/2 + A_22/2)**2 - A_11*A_22 - A_12*A_21
        damping_coeff = gamma/s**2
        
        half_time_list = []
        for sample in range(tf.shape(posterior)[0]):
            params = posterior[sample,:]
            half_time = self.calc_half_life(params)
            half_time_list.append(half_time)
        half_time = np.array(half_time_list)
        mean_heights = tf.reduce_mean(meal_heights,1)
        sum_heights = tf.reduce_sum(meal_heights,1)
    
        posterior_params_dict = {}
        variable_names = ['A_0', 'A_1', 'phi_1', 'A_11', 'A_12', 'A_21', 'A_22', 
                          'tau', 'B_22', 'sigma', 'meal_heights','damping_coeff',
                          'half_time','mean_heights','sum_heights']  
        for variable in variable_names:
            posterior_params_dict[variable] = eval(variable)
            
        self.posterior_params_dict = posterior_params_dict
    
    def calc_half_life(self, params):
        """Calculates the half life of the glucose response function, when
            a meal causes a glucose increase of 1 mmol/L
        
        :param params: an unconstrained TensorFlow array of parameter values
        :return: the half time of the glucose response function (hours)
        """
        t = np.linspace(0,15,1000)  
        A_0, A_1, phi_1, A_11, A_12, A_21, A_22, tau, B_11, B_22, sigma, meal_heights1, meal_heights2 = self.transform_params(params)
        s = -(A_11+A_22)/2
        gamma = (A_11/2 + A_22/2)**2 - A_11*A_22 - A_12*A_21
        
        normalisation_factor = tf.cond(tf.math.greater_equal(gamma, 0),
                 lambda : (A_21*tf.exp(tf.sqrt(gamma)*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)+s*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)))/(2*tf.sqrt(gamma)) - (A_21*tf.exp(-tf.sqrt(gamma)*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)+s*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)))/(2*tf.sqrt(gamma)),
                 lambda : (A_21*tf.exp(s*tf.atan(-tf.sqrt(-gamma)/s)/tf.sqrt(-gamma))*tf.sin(tf.sqrt(-gamma)*tf.atan(-tf.sqrt(-gamma)/s)/tf.sqrt(-gamma)))/tf.sqrt(-gamma))
        
        height = 1/normalisation_factor
    
        perturbation_fn = tf.cond(tf.math.greater_equal(gamma, 0),
                             lambda : (A_21*tf.exp(tf.sqrt(gamma)*t+s*t))/(2*tf.sqrt(gamma)) - (A_21*tf.exp(-tf.sqrt(gamma)*t+s*t))/(2*tf.sqrt(gamma)),
                             lambda : (A_21*tf.exp(s*t)*tf.sin(tf.sqrt(-gamma)*t))/tf.sqrt(-gamma))
        
        y_mean = height*perturbation_fn
        peak_time = t[np.argmax(y_mean)]
        t_after = t[t>peak_time]
        y_filt_after = y_mean[t>peak_time]
        half_value = np.max(y_mean)/2
        half_time = t_after[np.argmin(y_filt_after>half_value)]
        
        return half_time
    
    def model_fit_metrics(self,samples_list):
        """ Starting from a list of posterior parameter samples, this return
            a dictionary of model fit metrics, including variance explained
            
        :param samples_list: a list of samples from the posterior parameter
            distribution
        :return: a dictionary containing model fit summary metrics
        """
        posterior = tf.concat(samples_list,0) 
        
        rval_circ_and_meals_list = []
        rval_meals_list = []
        expl_var_circ_and_meals_list = []
        expl_var_meals_list = []
        
        for sample in range(tf.shape(posterior)[0]):
            params = posterior[sample,:]
            rval_circ_and_meals, rval_meals, expl_var_circ_and_meals, expl_var_meals = self.get_model_fit(params)
            rval_circ_and_meals_list.append(rval_circ_and_meals)
            rval_meals_list.append(rval_meals)
            expl_var_circ_and_meals_list.append(expl_var_circ_and_meals)
            expl_var_meals_list.append(expl_var_meals)
        
        model_fit_dict = {'rval_circ_and_meals':np.array(rval_circ_and_meals_list),
                                 'rval_meals':np.array(rval_meals_list),
                                 'expl_var_circ_and_meals':np.array(expl_var_circ_and_meals_list),
                                 'expl_var_meals':np.array(expl_var_meals_list)}
        
        self.model_fit_dict = model_fit_dict
    
    def get_model_fit(self, params):
        """Calculates the model fit summary metrics from a single
            TensorFlow array of parameter values
        
        :param params: a TensorFlow array of parameter values
        :return: summary metrics of model fit
        """

        circ_and_meals1, meals1, circ1 = self.all_mean_funcs(params,self.X1,
                                                             self.T_sub_timestamp1,
                                                             self.food_item_index1,1)
        circ_and_meals2, meals2, circ2 = self.all_mean_funcs(params,self.X2,
                                                             self.T_sub_timestamp2,
                                                             self.food_item_index2,2)
        
        Y = tf.concat((self.Y1,self.Y2),0)
        circ_and_meals = tf.concat((circ_and_meals1,circ_and_meals2),0)
        meals = tf.concat((meals1,meals2),0)

        rval_meals, pval = pearsonr(meals.numpy().reshape(-1),Y.numpy().reshape(-1))
        rval_circ_and_meals, pval = pearsonr(circ_and_meals.numpy().reshape(-1),Y.numpy().reshape(-1))
        
        var_data = np.var(Y)
        
        residual_meals = Y-meals
        var_residual_meals = np.var(residual_meals)
        expl_var_meals = 1-var_residual_meals/var_data
         
        residual_circ_and_meals = Y-circ_and_meals
        var_residual_circ_and_meals = np.var(residual_circ_and_meals)
        expl_var_circ_and_meals = 1-var_residual_circ_and_meals/var_data
            
        return rval_circ_and_meals, rval_meals, expl_var_circ_and_meals, expl_var_meals


class Model1SingleSensor(MSSModel):
    """ Class that implements Model1, which models glucose CGM and
        meal timestamp data. This class is for the case when there is one
        CGM sensor. The model is based on a 2-D system of
        stochastic differential equations, which is implemented using the
        TensorFlow Probability distribution LinearGaussianStateSpaceModel. 
        
        :param DatasetModel1Concatenated: a DatasetModel1Concatenated class
            that contains all data for the CGM sensors and meal timestamps
    """
    def __init__(
            self,
            DatasetModel1Concatenated,
        ) -> None:
        
        self.__dict__ = DatasetModel1Concatenated.__dict__
        
        self.epsilon = 1e-6
        self.w = tf.constant(2*np.pi/24)
        self.prior = tfd.JointDistributionSequentialAutoBatched([
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[5.]),
                                         bijector=tfb.Log(), name='A_0'),
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),
                                         bijector=tfb.Log(), name='A_1'),
             tfd.VonMises(loc=[0.0], concentration=[0.0], name='phi_1'),
             tfp.distributions.Normal(loc = [0.0],scale = [0.5], name='A_11'),
             tfp.distributions.Normal(loc = [0.0],scale = [0.5], name='A_12'),
             tfp.distributions.Normal(loc = [0.0],scale = [0.5], name='A_21'),   
             tfp.distributions.Normal(loc = [0.0],scale = [0.5], name='A_22'),      
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.5]),
                                         bijector=tfb.Log(), name='tau'),
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.5]),
                                         bijector=tfb.Log(), name='B_11'),
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.5]),
                                         bijector=tfb.Log(), name='B_22'),
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=5*tf.ones(self.N_meals1,dtype=tf.float32)),
                                         bijector=tfb.Log(), name='meal_heights1'),
             ]) 

        super().__init__()
                
    def transform_params(self,params):
        """ Transforms an unconstrained TensorFlow array into named variables
            with constraints applied
            
        :param params: an unconstrained TensorFlow array of parameter values
        :return: named parameter variables with constraints applied
        """
        A_0 = tf.math.exp(params[0,tf.newaxis])
        A_1 = tf.math.exp(params[1,tf.newaxis])
        phi_1 = params[2,tf.newaxis]
        A_11 = tf.math.exp(params[3,tf.newaxis])
        A_21 = tf.math.exp(params[5,tf.newaxis])
        A_12 = tf.math.exp(params[4,tf.newaxis])
        A_22 = tf.math.exp(params[6,tf.newaxis])
        tau = tf.math.exp(params[7,tf.newaxis])
        B_11 = 0.#tf.math.exp(params[7,tf.newaxis])
        B_22 = tf.math.exp(params[8,tf.newaxis])
        sigma = tf.math.exp(params[9,tf.newaxis]) 
        meal_heights1 = tf.math.exp(params[self.N_global:,tf.newaxis])

        return A_0, A_1, phi_1, A_11, A_12, A_21, A_22, tau, B_11, B_22, sigma, meal_heights1
        
    def sample_prior(self):
        """ Draws a sample from the model prior, returned as a TensorFlow array
        
        :returns: a real Tensor 
        :rtype: tf.float32
        """    
        
        prior_sample = self.prior.sample()
        sample_vector = tf.concat(prior_sample,0)
        return sample_vector  
    
    def unpack_vector(self,params):
        """ Converts from an unconstrained TensorFlow array to a list, 
            which is compatible with the TensorFlow Probability 
            distribution JointDistributionSequentialAutoBatched
            
        :param params: an unconstrained TensorFlow array of parameter values
        :return: list of parameter variables
        """
        params_unpacked = [params[0,tf.newaxis],
                           params[1,tf.newaxis],
                           params[2,tf.newaxis],
                           params[3,tf.newaxis],
                           params[4,tf.newaxis],
                           params[5,tf.newaxis],
                           params[6,tf.newaxis],
                           params[7,tf.newaxis],
                           params[8,tf.newaxis],
                           params[9,tf.newaxis],
                           params[self.N_global:]]
        return params_unpacked
    
    def mean_meals(self,A_0,A_1,phi_1,s,gamma,A_21,tau,meal_heights,X,T_sub_timestamp,food_item_index):
        """ Using the parameters of the model and information on the timestamps
            of recorded meals, this function returns the meal function r(t)
            
        :param A_0: the baseline glucose value
        :param A_1: the amplitude of the oscillation
        :param phi_1: the peak time of the oscillation
        :param s: parameter related to the eigenvalues of the Model1 A matrix
        :param gamma: parameter related to the eigenvalues of the Model1 A matrix
        :param A_21: of the Model1 A matrix
        :param tau: the time delay between meal ingestion and the start of
            the response function
        :param meal_heights: the heights of the meal response function. There
            is one height associated with each meal
        :param X: the time vector of input glucose CGM data
        :param T_sub_timestamp: an array that subtracts the recorded meal times
            from the time vector X
        :param food_item_index: a list of integers that control if meals share
            the same height response function. This is the case when meals
            have the same free text annotation
        :return: the meal response function    
        """
        normalisation_factor = tf.cond(tf.math.greater_equal(gamma, 0),
                     lambda : (A_21*tf.exp(tf.sqrt(gamma)*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)+s*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)))/(2*tf.sqrt(gamma)) - (A_21*tf.exp(-tf.sqrt(gamma)*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)+s*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)))/(2*tf.sqrt(gamma)),
                     lambda : (A_21*tf.exp(s*tf.atan(-tf.sqrt(-gamma)/s)/tf.sqrt(-gamma))*tf.sin(tf.sqrt(-gamma)*tf.atan(-tf.sqrt(-gamma)/s)/tf.sqrt(-gamma)))/tf.sqrt(-gamma))
    
        meal_heights = meal_heights/normalisation_factor
            
        T_heaviside = tf.experimental.numpy.heaviside(T_sub_timestamp-tau,1)
        t = T_heaviside*(T_sub_timestamp-tau)
        
        perturbation_fn = tf.cond(tf.math.greater_equal(gamma, 0),
                             lambda : (A_21*tf.exp(tf.sqrt(gamma)*t+s*t))/(2*tf.sqrt(gamma)) - (A_21*tf.exp(-tf.sqrt(gamma)*t+s*t))/(2*tf.sqrt(gamma)),
                             lambda : (A_21*tf.exp(s*t)*tf.sin(tf.sqrt(-gamma)*t))/tf.sqrt(-gamma))
        
        heights_unpacked = tf.gather_nd(meal_heights,food_item_index)
                
        return A_0 + tf.reduce_sum(heights_unpacked*perturbation_fn,0)
    
    def mean_circadian(self,A_0,A_1,phi_1,X):
        """ The circadian rhythm in glucose values
        
        :param A_0: the baseline glucose value
        :param A_1: the amplitude of the oscillation
        :param phi_1: the peak time of the oscillation
        :return: the circadian function
        """
        return A_0 + A_1*(1+tf.cos(self.w*X-phi_1))/2 
    
    def mean_meals_and_circadian(self,A_0,A_1,phi_1,s,gamma,A_21,tau,
                                 meal_heights,X,T_sub_timestamp,food_item_index):
        """ Using the parameters of the model and information on the timestamps
            of recorded meals, this function returns the sum of the
            meal function and the circadian baseline oscillation
            
        :param A_0: the baseline glucose value
        :param A_1: the amplitude of the oscillation
        :param phi_1: the peak time of the oscillation
        :param s: parameter related to the eigenvalues of the Model1 A matrix
        :param gamma: parameter related to the eigenvalues of the Model1 A matrix
        :param A_21: of the Model1 A matrix
        :param tau: the time delay between meal ingestion and the start of
            the response function
        :param meal_heights: the heights of the meal response function. There
            is one height associated with each meal
        :param X: the time vector of input glucose CGM data
        :param T_sub_timestamp: a matrix that subtracts the recorded meal times
            from the time vector X
        :param food_item_index: a list of integers that control if meals share
            the same height response function. This is the case when meals
            have the same free text annotation
        :return: the sum of the meal response function and the circadian baseline    
        """    
        f_meals = self.mean_meals(A_0,A_1,phi_1,s,gamma,A_21,tau,
                                  meal_heights,X,T_sub_timestamp,food_item_index)
        f_circadian = self.mean_circadian(A_0,A_1,phi_1,X)
        
        return f_meals+f_circadian-A_0
    
    def all_mean_funcs(self,params,X,T_sub_timestamp,food_item_index,sensor):
        """ Using the parameters of the model and information on the timestamps
            of recorded meals, this function returns the sum of the
            meal function and the circadian baseline oscillation
            
        :param params: an unconstrained TensorFlow array of parameter values
        :param X: the time vector of input glucose CGM data
        :param T_sub_timestamp: a matrix that subtracts the recorded meal times
            from the time vector X
        :param food_item_index: a list of integers that control if meals share
            the same height response function. This is the case when meals
            have the same free text annotation
        :return: the meal response function, the circadian baseline and the sum 
            of the meal response function and the circadian baseline    
        """  
        
        A_0, A_1, phi_1, A_11, A_12, A_21, A_22, tau, B_11, B_22, sigma, meal_heights = self.transform_params(params)

        s = -(A_11+A_22)/2
        gamma = (A_11/2 + A_22/2)**2 - A_11*A_22 - A_12*A_21
        
        f_meals_and_circadian = self.mean_meals_and_circadian(A_0,A_1,phi_1,s,gamma,
                                                              A_21,tau,meal_heights,
                                                              X,T_sub_timestamp,
                                                              food_item_index)
        f_meals = self.mean_meals(A_0,A_1,phi_1,s,gamma,A_21,tau,
                                  meal_heights,X,T_sub_timestamp,food_item_index)
        f_circadian = self.mean_circadian(A_0,A_1,phi_1,X)
        
        return f_meals_and_circadian, f_meals, f_circadian
      
    def model1_ssm(self,params,X,Y,T_sub_timestamp,food_item_index):
        """ Creates Model1 as a TensorFlow Probability 
        LinearGaussianStateSpaceModel distribution. The log liklihood
        of this model is subsequently used for parameter inference
            
        :param params: an unconstrained TensorFlow array of parameter values
        :param X: the time vector of input glucose CGM data (in absolute hours)
        :param T_sub_timestamp: an array that subtracts the recorded meal times
            from the time vector X
        :param food_item_index: a list of integers that control if meals share
            the same height response function. This is the case when meals
            have the same free text annotation
        :return: Model1 as a LinearGaussianStateSpaceModel object   
        """ 
        
        Delta_t = X[1]-X[0]
        A_0, A_1, phi_1, A_11, A_12, A_21, A_22, tau, B_11, B_22, sigma, meal_heights = self.transform_params(params)
        
        s = -(A_11+A_22)/2
        gamma = (A_11/2 + A_22/2)**2 - A_11*A_22 - A_12*A_21
        
        meanf = self.mean_meals_and_circadian(A_0,A_1,phi_1,s,gamma,A_21,tau,
                                              meal_heights,X,T_sub_timestamp,
                                              food_item_index)
    
        def f_expm_M(s,gamma):
          return tf.cond(tf.math.greater_equal(gamma, 0),
                         lambda : tf.concat([tf.stack([-tf.exp(s*Delta_t)*((A_11*tf.sinh(tf.sqrt(gamma)*Delta_t))/tf.sqrt(gamma) - tf.cosh(tf.sqrt(gamma)*Delta_t) + (s*tf.sinh(tf.sqrt(gamma)*Delta_t))/tf.sqrt(gamma)),-(A_12*tf.exp(s*Delta_t)*tf.sinh(tf.sqrt(gamma)*Delta_t))/tf.sqrt(gamma)], 1),tf.stack([(A_21*tf.exp(s*Delta_t)*tf.sinh(tf.sqrt(gamma)*Delta_t))/tf.sqrt(gamma), -tf.exp(s*Delta_t)*((A_22*tf.sinh(tf.sqrt(gamma)*Delta_t))/tf.sqrt(gamma) - tf.cosh(tf.sqrt(gamma)*Delta_t) + (s*tf.sinh(tf.sqrt(gamma)*Delta_t))/tf.sqrt(gamma))], 1)],0),
                         lambda : tf.concat([tf.stack([-tf.exp(s*Delta_t)*((A_11*tf.sin(tf.sqrt(-gamma)*Delta_t))/tf.sqrt(-gamma) - tf.cos(tf.sqrt(-gamma)*Delta_t) + (s*tf.sin(tf.sqrt(-gamma)*Delta_t))/tf.sqrt(-gamma)),-(A_12*tf.exp(s*Delta_t)*tf.sin(tf.sqrt(-gamma)*Delta_t))/tf.sqrt(-gamma)], 1),tf.stack([(A_21*tf.exp(s*Delta_t)*tf.sin(tf.sqrt(-gamma)*Delta_t))/tf.sqrt(-gamma), -tf.exp(s*Delta_t)*((A_22*tf.sin(tf.sqrt(-gamma)*Delta_t))/tf.sqrt(-gamma) - tf.cos(tf.sqrt(-gamma)*Delta_t) + (s*tf.sin(tf.sqrt(-gamma)*Delta_t))/tf.sqrt(-gamma))], 1)],0))
    
        expm_M = f_expm_M(s,gamma)
        P = tf.concat([tf.stack([(B_11*A_11**2*A_22 + A_21*B_11*A_11*A_12 + B_11*A_11*A_22**2 + B_22*A_12**2*A_22)/((A_11*A_22 + A_12*A_21)*(A_11 + A_22)),(A_11*A_22*(A_21*B_11 - A_12*B_22))/((A_11*A_22 + A_12*A_21)*(A_11 + A_22))], 1),tf.stack([(A_11*A_22*(A_21*B_11 - A_12*B_22))/((A_11*A_22 + A_12*A_21)*(A_11 + A_22)), (B_22*A_11**2*A_22 + B_11*A_11*A_21**2 + B_22*A_11*A_22**2 + A_12*B_22*A_21*A_22)/((A_11*A_22 + A_12*A_21)*(A_11 + A_22))], 1)],0)
        initial_loc = tf.stack([0.,0.],0)
        initial_scale_diag = tf.concat([(B_11*A_11**2*A_22 + A_21*B_11*A_11*A_12 + B_11*A_11*A_22**2 + B_22*A_12**2*A_22)/((A_11*A_22 + A_12*A_21)*(A_11 + A_22)),(B_22*A_11**2*A_22 + B_11*A_11*A_21**2 + B_22*A_11*A_22**2 + A_12*B_22*A_21*A_22)/((A_11*A_22 + A_12*A_21)*(A_11 + A_22))],0)        
        transition_cov = P - tf.linalg.matmul(expm_M,tf.linalg.matmul(P,tf.transpose(expm_M)))+ tf.linalg.diag([self.epsilon,self.epsilon])  
          
        transition_matrix = expm_M
        transition_noise=tfd.MultivariateNormalTriL(
            scale_tril=tf.linalg.cholesky(transition_cov))
        
        def observation_noise(t):
            loc = meanf[t]
            return tfd.MultivariateNormalDiag(
                    loc=loc,
                    scale_diag=sigma**2 * tf.ones([1]))        
        
        model = tfd.LinearGaussianStateSpaceModel(
              num_timesteps=len(X),
              transition_matrix=transition_matrix,
              transition_noise=transition_noise,
              observation_matrix=tf.stack([[0., 1.]], 0),
               observation_noise=observation_noise,
              initial_state_prior=tfd.MultivariateNormalDiag(
               loc =  initial_loc,     
               scale_diag=initial_scale_diag))
        
        return model    
    
    def MAP_model1_ssm(self,params):
        """ The negative of the unnormalized posterior distribution. The 
        negative is taken as the optimisation algorithm will minimise the
        objective function i.e. it will find the maximum of the 
        unnormalized posterior distribution (the MAP)
        
        :param params: an unconstrained TensorFlow array of parameter values
        :return: the negative of the unnormalized posterior distribution
        """
        
        return -self.unnormalized_posterior(params) 
    
    def unnormalized_posterior(self,params):
        """ The unnormalized posterior distribution. This is used for MCMC
        sampling
        
        :param params: an unconstrained TensorFlow array of parameter values
        :return: the unnormalized posterior distribution
        """
        unpacked_vector = self.unpack_vector(params)
        loss_prior = self.prior.log_prob(unpacked_vector)
        
        model1 = self.model1_ssm(params,self.X1,self.Y1,self.T_sub_timestamp1,
                                 self.food_item_index1)
        loss_time_series1 = model1.log_prob(self.Y1)
           
        return loss_time_series1 + loss_prior   
    
    def lik_model1_ssm(self,params):
        """ The log probability density of Model1
        
        :param params: an unconstrained TensorFlow array of parameter values
        :return: the log probability density of Model1
        """
        model1 = self.model1_ssm(params,self.X1,self.Y1,self.T_sub_timestamp1,
                                 self.food_item_index1)
        loss_time_series1 = model1.log_prob(self.Y1)
           
        return loss_time_series1 
    
    @tf.function(jit_compile=True)
    def get_loss_and_grads_map(self,params):
        """The loss and gradients of the Model1 MAP function, which are 
        needed by the tfp.optimizer.bfgs_minimize algorithm used for
        finding the MAP solution
        
        :param params: an unconstrained TensorFlow array of parameter values
        :return: the loss and gradients of the MAP function
        """
        with tf.GradientTape() as tape:
            tape.watch(params)
            loss = self.MAP_model1_ssm(params)
        grads = tape.gradient(loss,params)
        return loss, grads
    
    def check_meals_glucose_fit(self,params,legend = True):
        """ For a given set of parameters, this function will compare the
        mean of Model1 (using the sum of the circadian and meal functions)
        with the data
        
        """

        f_meals_and_circadian, _, _ = self.all_mean_funcs(params,
                                                          self.X1,
                                                          self.T_sub_timestamp1,
                                                          self.food_item_index1,
                                                          1)
        
        plt.plot(self.X1,f_meals_and_circadian,color=config.COLOR_PRED1,
                                 linewidth=1,label='Meal+circ. model')
        plt.plot(self.X1,self.Y1,color=config.COLOR_GLUC,
                                 linewidth=1,label='CGM data')
        
        for timestamp in self.timestamps1:
            plt.plot([timestamp,timestamp],[2,10],'k:',alpha=0.1,linewidth=1)
        if legend:
            plt.legend()
        rval, pval = pearsonr(f_meals_and_circadian,self.Y1[:,0])
        plt.title('R = %(correlation)1.2f' % {"correlation": rval})
        plt.xlabel('Time (h)')
        plt.ylabel('Glucose (mmol/L)')
        

    def transform_posterior_params(self,samples_list):
        """ Creates a dictionary of posterior parameter samples for named
            parameters
            
        :param samples_list: a list of samples from the posterior parameter
            distribution
        :return: a dictionary that contains samples of the the posterior
            distribution for named parameters
        """
        posterior = tf.concat(samples_list,0) 
        
        A_0 = tf.math.exp(posterior[:,0])
        A_1 = tf.math.exp(posterior[:,1])
        phi_1 = posterior[:,2]
        A_11 = tf.math.exp(posterior[:,3])
        A_21 = tf.math.exp(posterior[:,5])
        A_12 = tf.math.exp(posterior[:,4])
        A_22 = tf.math.exp(posterior[:,6])
        tau = tf.math.exp(posterior[:,7])
        B_22 = tf.math.exp(posterior[:,8])
        sigma = tf.math.exp(posterior[:,9])
        meal_heights = tf.math.exp(posterior[:,self.N_global:])
        
        s = -(A_11+A_22)/2
        gamma = (A_11/2 + A_22/2)**2 - A_11*A_22 - A_12*A_21
        damping_coeff = gamma/s**2
        
        half_time_list = []
        for sample in range(tf.shape(posterior)[0]):
            params = posterior[sample,:]
            half_time = self.calc_half_life(params)
            half_time_list.append(half_time)
        half_time = np.array(half_time_list)
        mean_heights = tf.reduce_mean(meal_heights,1)
        sum_heights = tf.reduce_sum(meal_heights,1)
    
        posterior_params_dict = {}
        variable_names = ['A_0', 'A_1', 'phi_1', 'A_11', 'A_12', 'A_21', 'A_22', 
                          'tau', 'B_22', 'sigma', 'meal_heights','damping_coeff',
                          'half_time','mean_heights','sum_heights']  
        for variable in variable_names:
            posterior_params_dict[variable] = eval(variable)
            
        self.posterior_params_dict = posterior_params_dict
    
    def calc_half_life(self, params):
        """Calculates the half life of the glucose response function, when
            a meal causes a glucose increase of 1 mmol/L
        
        :param params: an unconstrained TensorFlow array of parameter values
        :return: the half time of the glucose response function (hours)
        """
        t = np.linspace(0,15,1000)  
        A_0, A_1, phi_1, A_11, A_12, A_21, A_22, tau, B_11, B_22, sigma, meal_heights = self.transform_params(params)
        s = -(A_11+A_22)/2
        gamma = (A_11/2 + A_22/2)**2 - A_11*A_22 - A_12*A_21
        
        normalisation_factor = tf.cond(tf.math.greater_equal(gamma, 0),
                 lambda : (A_21*tf.exp(tf.sqrt(gamma)*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)+s*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)))/(2*tf.sqrt(gamma)) - (A_21*tf.exp(-tf.sqrt(gamma)*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)+s*tf.atanh(-tf.sqrt(gamma)/s)/tf.sqrt(gamma)))/(2*tf.sqrt(gamma)),
                 lambda : (A_21*tf.exp(s*tf.atan(-tf.sqrt(-gamma)/s)/tf.sqrt(-gamma))*tf.sin(tf.sqrt(-gamma)*tf.atan(-tf.sqrt(-gamma)/s)/tf.sqrt(-gamma)))/tf.sqrt(-gamma))
        
        height = 1/normalisation_factor
    
        perturbation_fn = tf.cond(tf.math.greater_equal(gamma, 0),
                             lambda : (A_21*tf.exp(tf.sqrt(gamma)*t+s*t))/(2*tf.sqrt(gamma)) - (A_21*tf.exp(-tf.sqrt(gamma)*t+s*t))/(2*tf.sqrt(gamma)),
                             lambda : (A_21*tf.exp(s*t)*tf.sin(tf.sqrt(-gamma)*t))/tf.sqrt(-gamma))
        
        y_mean = height*perturbation_fn
        peak_time = t[np.argmax(y_mean)]
        t_after = t[t>peak_time]
        y_filt_after = y_mean[t>peak_time]
        half_value = np.max(y_mean)/2
        half_time = t_after[np.argmin(y_filt_after>half_value)]
        
        return half_time
    
    def model_fit_metrics(self,samples_list):
        """ Starting from a list of posterior parameter samples, this return
            a dictionary of model fit metrics, including variance explained
            
        :param samples_list: a list of samples from the posterior parameter
            distribution
        :return: a dictionary containing model fit summary metrics
        """
        posterior = tf.concat(samples_list,0) 
        
        rval_circ_and_meals_list = []
        rval_meals_list = []
        expl_var_circ_and_meals_list = []
        expl_var_meals_list = []
        
        for sample in range(tf.shape(posterior)[0]):
            params = posterior[sample,:]
            rval_circ_and_meals, rval_meals, expl_var_circ_and_meals, expl_var_meals = self.get_model_fit(params)
            rval_circ_and_meals_list.append(rval_circ_and_meals)
            rval_meals_list.append(rval_meals)
            expl_var_circ_and_meals_list.append(expl_var_circ_and_meals)
            expl_var_meals_list.append(expl_var_meals)
        
        model_fit_dict = {'rval_circ_and_meals':np.array(rval_circ_and_meals_list),
                                 'rval_meals':np.array(rval_meals_list),
                                 'expl_var_circ_and_meals':np.array(expl_var_circ_and_meals_list),
                                 'expl_var_meals':np.array(expl_var_meals_list)}
        
        self.model_fit_dict = model_fit_dict
    
    def get_model_fit(self, params):
        """Calculates the model fit summary metrics from a single
            TensorFlow array of parameter values
        
        :param params: a TensorFlow array of parameter values
        :return: summary metrics of model fit
        """

        circ_and_meals, meals, circ = self.all_mean_funcs(params,self.X1,
                                                             self.T_sub_timestamp1,
                                                             self.food_item_index1,
                                                             1)
        Y = self.Y1[:,0]

        rval_meals, pval = pearsonr(meals.numpy().reshape(-1),Y.numpy().reshape(-1))
        rval_circ_and_meals, pval = pearsonr(circ_and_meals.numpy().reshape(-1),Y.numpy().reshape(-1))
        
        var_data = np.var(Y)
        
        residual_meals = Y-meals
        var_residual_meals = np.var(residual_meals)
        expl_var_meals = 1-var_residual_meals/var_data
         
        residual_circ_and_meals = Y-circ_and_meals
        var_residual_circ_and_meals = np.var(residual_circ_and_meals)
        expl_var_circ_and_meals = 1-var_residual_circ_and_meals/var_data
            
        return rval_circ_and_meals, rval_meals, expl_var_circ_and_meals, expl_var_meals
   

class Model1SingleSensor_nocirc(Model1SingleSensor):
    """ Class that implements Model1 (which models glucose CGM and
        meal timestamp data), except the underlying 24-hour mean baseline 
        trend is removed. NOTE that MCMC will not work correctly for this
        class, as it is designed for simple testing with MAP optimisation
        
        :param DatasetModel1Concatenated: a DatasetModel1Concatenated class
            that contains all data for the CGM sensors and meal timestamps
    """
    def __init__(
            self,
            DatasetModel1Concatenated,
        ) -> None:
        
        self.__dict__ = DatasetModel1Concatenated.__dict__
        
        self.epsilon = 1e-6
        self.w = tf.constant(2*np.pi/24)
        self.prior = tfd.JointDistributionSequentialAutoBatched([
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[5.]),
                                         bijector=tfb.Log(), name='A_0'),
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),
                                         bijector=tfb.Log(), name='A_1'),
             tfd.VonMises(loc=[0.0], concentration=[0.0], name='phi_1'),
             tfp.distributions.Normal(loc = [0.0],scale = [0.5], name='A_11'),
             tfp.distributions.Normal(loc = [0.0],scale = [0.5], name='A_12'),
             tfp.distributions.Normal(loc = [0.0],scale = [0.5], name='A_21'),   
             tfp.distributions.Normal(loc = [0.0],scale = [0.5], name='A_22'),      
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.5]),
                                         bijector=tfb.Log(), name='tau'),
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.5]),
                                         bijector=tfb.Log(), name='B_11'),
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.5]),
                                         bijector=tfb.Log(), name='B_22'),
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=5*tf.ones(self.N_meals1,dtype=tf.float32)),
                                         bijector=tfb.Log(), name='meal_heights1'),
             ]) 

        super().__init__(DatasetModel1Concatenated)
        
    def transform_params(self,params):
        A_0 = tf.math.exp(params[0,tf.newaxis])
        A_1 = 0
        phi_1 = params[2,tf.newaxis]
        A_11 = tf.math.exp(params[3,tf.newaxis])
        A_21 = tf.math.exp(params[5,tf.newaxis])
        A_12 = tf.math.exp(params[4,tf.newaxis])
        A_22 = tf.math.exp(params[6,tf.newaxis])
        gamma = tf.math.exp(params[7,tf.newaxis])
        B_11 = 0.
        B_22 = tf.math.exp(params[8,tf.newaxis])
        sigma = tf.math.exp(params[9,tf.newaxis]) 
        heights_ind1 = tf.math.exp(params[self.N_global:,tf.newaxis])

        return A_0, A_1, phi_1, A_11, A_12, A_21, A_22, gamma, B_11, B_22, sigma, heights_ind1
