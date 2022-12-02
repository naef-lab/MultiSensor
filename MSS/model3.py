#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 03:32:52 2022

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
from tensorflow_probability.python import bijectors as tfb
from scipy.stats.stats import pearsonr
from tensorflow_probability import distributions as tfd
from .abstractmodel import MSSModel
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

#%%

class Model3Integrated(MSSModel):
    """ Class that implements Model3, which models food, glucose,
        physical activity, heart rate (HR) and heart rate variability (HRV). 
        Model3 is based on a 5-D system of
        stochastic differential equations, which is implemented using the
        TensorFlow Probability distribution LinearGaussianStateSpaceModel. 
        
        :param DatasetModel3Integrated: a DatasetModel2Actiheart class
            that contains data from the Actiheart sensor
        :param acti_params: the parameters from Model2, which are fixed and
            not refitted
    """
    def __init__(
            self,
            DatasetModel3Integrated,acti_params
        ) -> None:

        self.__dict__ = DatasetModel3Integrated.__dict__
        
        self.acti_params = acti_params
        
        self.epsilon = 1e-3
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
             tfp.distributions.Normal(loc = [0.0],scale = [1.], name='C_51'), 
             tfp.distributions.Normal(loc = [0.0],scale = [1.], name='C_52'),
             tfp.distributions.Normal(loc = [0.0],scale = [1.], name='C_53'),
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.5]),
                                         bijector=tfb.Log(), name='tau'),                         
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.5]),
                                         bijector=tfb.Log(), name='B_11'),  
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.5]),
                                         bijector=tfb.Log(), name='B_22'), 
             tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=5*tf.ones(self.N_meals,dtype=tf.float32)),
                                         bijector=tfb.Log(), name='meal_heights'),                   
             ])   

        super().__init__()  
    
    def sample_prior(self):
        """ Draws a sample from the model prior, returned as a TensorFlow array
        
        :returns: a Tensor array
        :rtype: tf.float32
        """    
        
        prior_sample = self.prior.sample()
        sample_vector = tf.concat(prior_sample,0)
        return sample_vector  
            
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
        C_51 = params[7,tf.newaxis]
        C_52 = params[8,tf.newaxis]
        C_53 = params[9,tf.newaxis]
        tau = tf.math.exp(params[10,tf.newaxis])
        B_11 = 0.
        B_22 = tf.math.exp(params[11,tf.newaxis])
        sigma = tf.math.exp(params[12,tf.newaxis])
            
        meal_heights = tf.math.exp(params[13:,tf.newaxis])
        
        return A_0, A_1, phi_1, A_11, A_12, A_21, A_22, C_51, C_52, C_53, tau, B_11, B_22, sigma, meal_heights
            
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
                           params[10,tf.newaxis],
                           params[11,tf.newaxis],
                           params[12,tf.newaxis],
                           params[13:]]
        return params_unpacked
    
    def transform_params_acti(self,params):
        """ Transforms an unconstrained TensorFlow array  representing
            the Model2 parameters into named variables
            with constraints applied
            
        :param params: an unconstrained TensorFlow array of parameter values
        :return: named parameter variables with constraints applied
        """
        C_0_act = tf.math.exp(params[0,tf.newaxis])
        C_1_act = tf.math.exp(params[1,tf.newaxis])
        phi_act = params[2]     
        C_11 = tf.math.exp(params[3,tf.newaxis])
        
        C_0_bpm = tf.math.exp(params[4,tf.newaxis]) 
        C_1_bpm = tf.math.exp(params[5,tf.newaxis]) 
        phi_bpm = params[6] 
        C_21 = tf.math.exp(params[7,tf.newaxis]) 
        C_22 = tf.math.exp(params[8,tf.newaxis]) 
        
        C_0_hrv = tf.math.exp(params[9,tf.newaxis]) 
        C_1_hrv = tf.math.exp(params[10,tf.newaxis]) 
        phi_hrv = params[11] 
        C_31 = tf.math.exp(params[12,tf.newaxis]) 
        C_33 = tf.math.exp(params[13,tf.newaxis]) 
        
        D_11 = tf.math.exp(params[14,tf.newaxis])**2
        sigma_matrix = tf.linalg.diag(tf.math.exp(params[15:17]))
        cholesk = tfb.CorrelationCholesky().forward(params[17,tf.newaxis])
        corr = tfb.CholeskyOuterProduct().forward(cholesk)+tf.linalg.diag([0.001])
        cov = tf.matmul(sigma_matrix, tf.matmul(corr, sigma_matrix))
        D_22 = cov[0,0,tf.newaxis]
        D_23 = cov[0,1,tf.newaxis]
        D_33 = cov[1,1,tf.newaxis]
            
        sigma_act = tf.math.exp(params[18,tf.newaxis])
        sigma_bpm = tf.math.exp(params[19,tf.newaxis])
        sigma_hrv = tf.math.exp(params[20,tf.newaxis])
        
        return C_0_act, C_1_act, phi_act, C_11, C_0_bpm, C_1_bpm, phi_bpm, C_21, C_22, C_31, C_33, C_0_hrv, C_1_hrv, phi_hrv, C_31, C_33, D_11, D_22, D_23, D_33, sigma_act, sigma_bpm, sigma_hrv 
  
    
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
    
    def mean_circ(self,A_0,A_1,phi_1,X):
        """ The circ rhythm in glucose values
        
        :param A_0: the baseline glucose value
        :param A_1: the amplitude of the oscillation
        :param phi_1: the peak time of the oscillation
        :return: the circ function
        """
        return A_0 + A_1*(1+tf.cos(self.w*X-phi_1))/2 
    
    def mean_meals_and_circ(self,A_0,A_1,phi_1,s,gamma,A_21,tau,
                                 meal_heights,X,T_sub_timestamp,food_item_index):
        """ Using the parameters of the model and information on the timestamps
            of recorded meals, this function returns the sum of the
            meal function and the circ baseline oscillation
            
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
        :return: the sum of the meal response function and the circ baseline    
        """    
        f_meals = self.mean_meals(A_0,A_1,phi_1,s,gamma,A_21,tau,
                                  meal_heights,X,T_sub_timestamp,food_item_index)
        f_circ = self.mean_circ(A_0,A_1,phi_1,X)
        
        return f_meals+f_circ-A_0
    
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
        
        A_0, A_1, phi_1, A_11, A_12, A_21, A_22, C_51, C_52, C_53, tau, B_11, B_22, sigma, meal_heights = self.transform_params(params)
        s = -(A_11+A_22)/2
        gamma = (A_11/2 + A_22/2)**2 - A_11*A_22 - A_12*A_21
        
        f_meals_and_circ = self.mean_meals_and_circ(A_0,A_1,phi_1,s,gamma,
                                                              A_21,tau,meal_heights,
                                                              X,T_sub_timestamp,
                                                              food_item_index)
        f_meals = self.mean_meals(A_0,A_1,phi_1,s,gamma,A_21,tau,
                                  meal_heights,X,T_sub_timestamp,food_item_index)
        f_circ = self.mean_circ(A_0,A_1,phi_1,X)
        
        return f_meals_and_circ, f_meals, f_circ
    
    def expm_taylor(self,F):
        """ Function to perform matrix exponentiation. Based on a Taylor
            expansion.
        :param F: matrix to be exponentiated
        :return: the matrix exponential 
        """
        latent_size = tf.shape(F)[0]
        k = 5
        j = 1
        A = F/2**j
        T_k = tf.eye(latent_size)
        T_k += A
        matrix_power = A
        
        for i in range(2,k+1):
            matrix_power = tf.linalg.matmul(matrix_power, A)
            T_k += matrix_power/(tf.exp(tf.math.lgamma(i+1.)))
    
        expm_T_k = T_k
        
        for i in range(2**j-1):
            expm_T_k = tf.linalg.matmul(expm_T_k, T_k)
    
        return expm_T_k
    
    def get_transition_probs(self,params,X,T_sub_timestamp,food_item_index,C_51_only,C_52_only,C_53_only):
        """ Using the parameters of the model and time of day, this function 
            returns all of the elements required to construct the transition
            model of the state space model
            
        :param params: an unconstrained TensorFlow array of parameter values
        :param X: the time vector
        :param T_sub_timestamp: an array that subtracts the recorded meal times
            from the time vector X
        :param food_item_index: a list of integers that control if meals share
            the same height response function. This is the case when meals
            have the same free text annotation
        :param C_51_only: specifies whether C_51 is the only connection
            to the glucose model
        :param C_52_only: specifies whether C_52 is the only connection
            to the glucose model
        :param C_53_only: specifies whether C_53 is the only connection
            to the glucose model     
        :return: the mean function of each variable and all matrices required
            for the transition model for Model3
        """  
        
        delta_t = X[1]-X[0]
        C_0_act, C_1_act, phi_act, C_11, C_0_bpm, C_1_bpm, phi_bpm, C_21, C_22, C_31, C_33, C_0_hrv, C_1_hrv, phi_hrv, C_31, C_33, D_11, D_22, D_23, D_33, sigma_act, sigma_bpm, sigma_hrv = self.transform_params_acti(self.acti_params)
        A_0, A_1, phi_1, A_11, A_12, A_21, A_22, C_51, C_52, C_53, tau, B_11, B_22, sigma, meal_heights = self.transform_params(params)
        
        if C_51_only==True:
            C_52 = [0.]
            C_53 = [0.]
        elif C_52_only==True:
            C_51 = [0.]
            C_53 = [0.]    
        elif C_53_only==True:
            C_51 = [0.]
            C_52 = [0.]  
            
        initial_loc = tf.stack([0.,0.,0.,0.,0.],0)
        initial_scale_diag = tf.concat([D_11/(2*C_11),(D_22*C_11**2 + C_22*D_22*C_11 + D_11*C_21**2)/(2*C_11*C_22*(C_11 + C_22)),(D_33*C_11**2 + C_33*D_33*C_11 + D_11*C_31**2)/(2*C_11*C_33*(C_11 + C_33)),(B_11*A_11**2*A_22 + A_21*B_11*A_11*A_12 + B_11*A_11*A_22**2 + B_22*A_12**2*A_22)/((A_11*A_22 + A_12*A_21)*(A_11 + A_22)),(B_22*A_11**2*A_22 + B_11*A_11*A_21**2 + B_22*A_11*A_22**2 + A_12*B_22*A_21*A_22)/((A_11*A_22 + A_12*A_21)*(A_11 + A_22))],0)
        initial_scale_diag = tf.sqrt(initial_scale_diag)
        
        mean_act = C_0_act + C_1_act*(1+tf.cos(self.w*X-phi_act))/2
        mean_bpm = C_0_bpm + C_1_bpm*(1+tf.cos(self.w*X-phi_bpm))/2 
        mean_hrv = C_0_hrv + C_1_hrv*(1+tf.cos(self.w*X-phi_hrv))/2
         
        s = -(A_11+A_22)/2
        gamma = (A_11/2 + A_22/2)**2 - A_11*A_22 - A_12*A_21
        mean_gluc = self.mean_meals_and_circ(A_0,A_1,phi_1,s,gamma,A_21,tau,meal_heights,X,T_sub_timestamp,food_item_index)
        
        E = tf.concat([tf.stack([ -C_11, [0.], [0.], [0.], [0.]], 1),tf.stack([ C_21, -C_22, [0.], [0.], [0.]], 1),tf.stack([ C_31, [0.], -C_33, [0.], [0.]], 1),tf.stack([ [0.], [0.], [0.], -A_11, -A_12], 1),tf.stack([ C_51, C_52, C_53, A_21, -A_22], 1)],0)
        F = tf.concat([tf.stack([ D_11, [0.], [0.], [0.], [0.]], 1),tf.stack([ [0.], D_22, D_23, [0.], [0.]], 1),tf.stack([ [0.], D_23, D_33, [0.], [0.]], 1),tf.stack([ [0.], [0.], [0.], [1e-6], [0.]], 1),tf.stack([ [0.], [0.], [0.], [0.], B_22], 1)],0)
        latent_size = tf.shape(F)[0]
        O = tf.zeros((latent_size,latent_size))
        I = tf.eye(latent_size)
        G =  tf.concat([tf.concat([E,F],1),tf.concat([O,tf.transpose(-E)],1)],0)
        G_expm = self.expm_taylor(G*delta_t)
        K = tf.concat([O,I],0)
        selector = tf.concat([I,O],1)
        C = tf.linalg.matmul(selector,tf.linalg.matmul(G_expm,K))
        expm_M = self.expm_taylor(E*delta_t)
        
        transition_matrix = expm_M
        transition_cov = tf.linalg.matmul(C,tf.transpose(expm_M))+tf.linalg.diag([self.epsilon,self.epsilon,self.epsilon,self.epsilon,self.epsilon])

        return mean_act, mean_bpm, mean_hrv, mean_gluc, transition_matrix, transition_cov, initial_loc, initial_scale_diag
            
    def model3_ssm(self,params,X,Y,T_sub_timestamp,food_item_index,C_51_only=False,C_52_only=False,C_53_only=False):
        """ Creates Model3 as a TensorFlow Probability 
        LinearGaussianStateSpaceModel distribution. The log liklihood
        of this model is subsequently used for parameter inference
            
        :param params: an unconstrained TensorFlow array of parameter values
        :param X: the time vector of input glucose CGM data (in absolute hours)
        :param T_sub_timestamp: an array that subtracts the recorded meal times
            from the time vector X
        :param food_item_index: a list of integers that control if meals share
            the same height response function. This is the case when meals
            have the same free text annotation
        :param C_51_only: specifies whether C_51 is the only connection
            to the glucose model
        :param C_52_only: specifies whether C_52 is the only connection
            to the glucose model
        :param C_53_only: specifies whether C_53 is the only connection
            to the glucose model    
        :return: Model1 as a LinearGaussianStateSpaceModel object   
        """ 

        C_0_act, C_1_act, phi_act, C_11, C_0_bpm, C_1_bpm, phi_bpm, C_21, C_22, C_31, C_33, C_0_hrv, C_1_hrv, phi_hrv, C_31, C_33, D_11, D_22, D_23, D_33, sigma_act, sigma_bpm, sigma_hrv = self.transform_params_acti(self.acti_params)
        A_0, A_1, phi_1, A_11, A_12, A_21, A_22, C_51, C_52, C_53, tau, B_11, B_22, sigma, meal_heights = self.transform_params(params)
                       
        mean_act, mean_bpm, mean_hrv, mean_gluc, transition_matrix, transition_cov, initial_loc, initial_scale_diag = self.get_transition_probs(params,X,T_sub_timestamp,food_item_index,C_51_only,C_52_only,C_53_only)    
        meanf = tf.stack([mean_act,mean_bpm,mean_hrv,mean_gluc], 0)
        
        transition_noise=tfd.MultivariateNormalTriL(
            scale_tril=tf.linalg.cholesky(transition_cov))
        
        observation_matrix = tf.stack([[1., 0., 0.,0.,0.],[0., 1., 0., 0.,0.],[0., 0., 1., 0.,0.],[0., 0., 0., 0.,1.]], 0)#tf.linalg.diag([1.,0.,1.,1.])
            
        def observation_noise(t):
            loc = meanf[:,t]
            return tfd.MultivariateNormalDiag(
                    loc=loc,
                    scale_diag=tf.concat([sigma_act**2,sigma_bpm**2,sigma_hrv**2,sigma**2], 0))        
        
        model = tfd.LinearGaussianStateSpaceModel(
              num_timesteps=len(X),
              transition_matrix=transition_matrix,
              transition_noise=transition_noise,
              observation_matrix=observation_matrix,
               observation_noise=observation_noise,
              initial_state_prior=tfd.MultivariateNormalDiag(
                    loc=initial_loc,
                    scale_diag=initial_scale_diag))
        
        return model  

    def MAP_model3_ssm(self,params):
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
        
        model3 = self.model3_ssm(params,self.X,self.Y,self.T_sub_timestamp,self.food_item_index)
        loss_time_series = model3.log_prob(self.Y,mask=self.mask)
        
        return loss_time_series + loss_prior 
    
    @tf.function(jit_compile=True)
    def lik_model2_ssm(self,params):
        """ The log probability density of Model3
        
        :param params: an unconstrained TensorFlow array of parameter values
        :return: the log probability density of Model1
        """

        model3 = self.model3_ssm(params,self.X,self.Y,self.T_sub_timestamp,self.food_item_index)
        loss_time_series = model3.log_prob(self.Y,mask=self.mask)
        
        return loss_time_series  
    
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
            loss = self.MAP_model3_ssm(params)
        grads = tape.gradient(loss,params)
        return loss, grads
            
    def transform_posterior_params(self):
        """ Creates a dictionary of posterior parameter samples for named
            parameters
            
        :param samples_list: a list of samples from the posterior parameter
            distribution
        :return: a dictionary that contains samples of the the posterior
            distribution for named parameters
        """
        posterior = tf.concat(self.samples_list,0) 
        
        A_0 = tf.math.exp(posterior[:,0])
        A_1 = tf.math.exp(posterior[:,1])
        phi_1 = posterior[:,2]
        A_11 = tf.math.exp(posterior[:,3])
        A_21 = tf.math.exp(posterior[:,5])
        A_12 = tf.math.exp(posterior[:,4])
        A_22 = tf.math.exp(posterior[:,6])
        C_51 = posterior[:,7]
        C_52 = posterior[:,8]
        C_53 = posterior[:,9]
        tau = tf.math.exp(posterior[:,10])
        B_11 = 0.
        B_22 = tf.math.exp(posterior[:,11])
        sigma = tf.math.exp(posterior[:,12])
            
        meal_heights = tf.math.exp(posterior[:,13:])
    
        posterior_params_dict = {'A_0':A_0, 'A_1':A_1, 'phi_1':phi_1, 'A_11':A_11, 'A_12':A_12, 'A_21':A_21, 'A_22':A_22, 'C_51':C_51, 'C_52':C_52, 'C_53':C_53, 'tau':tau, 'B_11':B_11, 'B_22':B_22, 'sigma':sigma, 'meal_heights':meal_heights}
        self.posterior_params_dict = posterior_params_dict
        
    @tf.function(jit_compile=True)
    def filter_model_with_activity_hr_hrv(self,params,X,Y,mask,T_sub_timestamp,food_item_index,C_51_only=False,C_52_only=False,C_53_only=False):

        """ Uses a Kalman forward filter to predict activity, HR, HRV and 
            glucose using activity, HR and HRV data
            
        :param params: an unconstrained TensorFlow array of parameter values
        :param X: the time vector of input glucose CGM data (in absolute hours)
        :param T_sub_timestamp: an array that subtracts the recorded meal times
            from the time vector X
        :param food_item_index: a list of integers that control if meals share
            the same height response function. This is the case when meals
            have the same free text annotation
        :param C_51_only: specifies whether C_51 is the only connection
            to the glucose model
        :param C_52_only: specifies whether C_52 is the only connection
            to the glucose model
        :param C_53_only: specifies whether C_53 is the only connection
            to the glucose model    
        :return: Model1 as a LinearGaussianStateSpaceModel object   
        """ 

        C_0_act, C_1_act, phi_act, C_11, C_0_bpm, C_1_bpm, phi_bpm, C_21, C_22, C_31, C_33, C_0_hrv, C_1_hrv, phi_hrv, C_31, C_33, D_11, D_22, D_23, D_33, sigma_act, sigma_bpm, sigma_hrv = self.transform_params_acti(self.acti_params)
        A_0, A_1, phi_1, A_11, A_12, A_21, A_22, C_51, C_52, C_53, tau, B_11, B_22, sigma, meal_heights = self.transform_params(params)
                       
        mean_act, mean_bpm, mean_hrv, mean_gluc, transition_matrix, transition_cov, initial_loc, initial_scale_diag = self.get_transition_probs(params,X,T_sub_timestamp,food_item_index,C_51_only,C_52_only,C_53_only)    
        meanf = tf.stack([mean_act,mean_bpm,mean_hrv], 0)
        mean_gluc1 = tf.zeros_like(X)
        mean_tot = tf.stack([mean_act,mean_bpm,mean_hrv,mean_gluc1,mean_gluc], 0)
        
        transition_noise=tfd.MultivariateNormalTriL(
            scale_tril=tf.linalg.cholesky(transition_cov))
        
        observation_matrix = tf.stack([[1., 0., 0.,0.,0.],[0., 1., 0., 0.,0.],[0., 0., 1., 0.,0.]], 0)
        
        def observation_noise(t):
            loc = meanf[:,t]#tf.concat([meanf[0,t,None],meanf[2:4,t]],0)#meanf[:,t]
            return tfd.MultivariateNormalDiag(
                    loc=loc,
                    scale_diag=tf.concat([sigma_act**2,sigma_bpm**2,sigma_hrv**2], 0))        
        
        model = tfd.LinearGaussianStateSpaceModel(
              num_timesteps=len(X),
              transition_matrix=transition_matrix,
              transition_noise=transition_noise,
              observation_matrix=observation_matrix,
               observation_noise=observation_noise,
              initial_state_prior=tfd.MultivariateNormalDiag(
                    loc=initial_loc,
                    scale_diag=initial_scale_diag))
        
        _, filtered_means, filtered_covs, _, _, _, _ = model.forward_filter(Y,mask=mask)
        
        return filtered_means+tf.transpose(mean_tot), filtered_covs

    def model_fit_metrics(self):
        """ Starting from a list of posterior parameter samples, this returns
            a dictionary of model fit metrics, including variance explained
            for predicting glucose using activity, HR and HRV
            
        :param samples_list: a list of samples from the posterior parameter
            distribution
        :return: a dictionary containing model fit summary metrics
        """
    
        posterior = tf.concat(self.samples_list,0) 
        
        expl_var_circ_list, expl_var_circ_and_meals_list, expl_var_activity_list, expl_var_activity_hr_list, expl_var_activity_hr_hrv_list, expl_var_diff_list = ([] for i in range(6))
        
        for sample in range(tf.shape(posterior)[0]):
            params = posterior[sample,:]
            expl_var_circ, expl_var_circ_and_meals, expl_var_activity, expl_var_activity_hr, expl_var_activity_hr_hrv, expl_var_diff = self.get_model_fit(params)
            
            expl_var_circ_list.append(expl_var_circ)
            expl_var_circ_and_meals_list.append(expl_var_circ_and_meals)
            expl_var_activity_hr_hrv_list.append(expl_var_activity_hr_hrv)
            expl_var_diff_list.append(expl_var_diff)
        
        model_fit_dict = {'expl_var_circ':np.array(expl_var_circ_list),
                                 'expl_var_circ_and_meals':np.array(expl_var_circ_and_meals_list),
                                 'expl_var_activity_hr_hrv':np.array(expl_var_activity_hr_hrv_list),
                                 'expl_var_diff':np.array(expl_var_diff_list)}

        self.model_fit_dict = model_fit_dict
        
    def get_model_fit(self,params):
        """Calculates the model fit summary metrics from a single
            TensorFlow array of parameter values
        
        :param params: a TensorFlow array of parameter values
        :return: summary metrics of model fit
        """
        
        mean_gluc, f_meals, mean_circ = self.all_mean_funcs(params,self.X,
                                                             self.T_sub_timestamp,
                                                             self.food_item_index,
                                                             self.sensor)

        Y_no_gluc = self.Y[:,:3]
        filtered_means, filtered_covs = self.filter_model_with_activity_hr_hrv(params,
                                                                               self.X,
                                                                               Y_no_gluc,
                                                                               self.mask,
                                                                               self.T_sub_timestamp,
                                                                               self.food_item_index)
        
        mean_activity_hr_hrv = filtered_means[:,4]
        
        y_dat = self.Y.numpy()[~self.mask,3]
        pred_circ = mean_circ[~self.mask]
        pred_circ_and_meals = mean_gluc[~self.mask]
        pred_activity_hr_hrv = mean_activity_hr_hrv[~self.mask]
        
        var_data = np.var(y_dat)
        
        residual_circ = y_dat-pred_circ
        var_residual_circ = np.var(residual_circ)
        expl_var_circ = 1-var_residual_circ/var_data
         
        residual_circ_and_meals = y_dat-pred_circ_and_meals
        var_residual_circ_and_meals = np.var(residual_circ_and_meals)
        expl_var_circ_and_meals = 1-var_residual_circ_and_meals/var_data
            
        residual_activity_hr_hrv = y_dat-pred_activity_hr_hrv
        var_residual_activity_hr_hrv = np.var(residual_activity_hr_hrv)
        expl_var_activity_hr_hrv = 1-var_residual_activity_hr_hrv/var_data
        
        expl_var_diff = expl_var_activity_hr_hrv-expl_var_circ_and_meals

        return expl_var_circ, expl_var_circ_and_meals, expl_var_activity_hr_hrv, expl_var_diff

    def MAP_model3_C51_only(self,params):
                
        unpacked_vector = self.unpack_vector(params)
        loss_prior = self.prior.log_prob(unpacked_vector)
        
        model3 = self.model3_ssm(params,self.X,self.Y,self.T_sub_timestamp,self.food_item_index,C_51_only=True)
        loss_time_series = model3.log_prob(self.Y,mask=self.mask)
    
        return -loss_time_series -loss_prior  
    
    @tf.function(jit_compile=True)
    def get_loss_and_grads_map_C51_only(self,params):
        with tf.GradientTape() as tape:
            tape.watch(params)
            loss = self.MAP_model3_C51_only(params)
        grads = tape.gradient(loss,params)
        return loss, grads
    
    def find_MAP_C51_only(self,loss_and_grads,N_starts=500,seed=123,max_iterations=1000,max_line_search_iterations=500,initial_inverse_hessian_scale = 1e-4,print_on=True):
        tf.random.set_seed(seed)
        MAP_list = []
        param_list = []
        for i in range(N_starts):
            if print_on:
                print(i)
            
            start = self.sample_prior()
    
            optim_results = tfp.optimizer.bfgs_minimize(
              loss_and_grads,
              initial_position=start,
              max_iterations=max_iterations,
              max_line_search_iterations=max_line_search_iterations,
              initial_inverse_hessian_estimate = initial_inverse_hessian_scale*tf.eye(len(start))
              )
    
            MAP_list.append(optim_results.objective_value)
            param_list.append(optim_results.position)
        
        self.MAP_opted_C51_only = MAP_list[np.nanargmin(MAP_list)]
        self.params_opted_map_C51_only = param_list[np.nanargmin(MAP_list)]
   
    def MAP_model3_C52_only(self,params):
                
        unpacked_vector = self.unpack_vector(params)
        loss_prior = self.prior.log_prob(unpacked_vector)
        
        model3 = self.model3_ssm(params,self.X,self.Y,self.T_sub_timestamp,self.food_item_index,C_52_only=True)
        loss_time_series = model3.log_prob(self.Y,mask=self.mask)
    
        return -loss_time_series -loss_prior  
    
    @tf.function(jit_compile=True)
    def get_loss_and_grads_map_C52_only(self,params):
        with tf.GradientTape() as tape:
            tape.watch(params)
            loss = self.MAP_model3_C52_only(params)
        grads = tape.gradient(loss,params)
        return loss, grads
    
    def find_MAP_C52_only(self,loss_and_grads,N_starts=500,seed=123,max_iterations=1000,max_line_search_iterations=500,initial_inverse_hessian_scale = 1e-4,print_on=True):
        tf.random.set_seed(seed)
        MAP_list = []
        param_list = []
        for i in range(N_starts):
            if print_on:
                print(i)
            
            start = self.sample_prior()
    
            optim_results = tfp.optimizer.bfgs_minimize(
              loss_and_grads,
              initial_position=start,
              max_iterations=max_iterations,
              max_line_search_iterations=max_line_search_iterations,
              initial_inverse_hessian_estimate = initial_inverse_hessian_scale*tf.eye(len(start))
              )
    
            MAP_list.append(optim_results.objective_value)
            param_list.append(optim_results.position)
        
        self.MAP_opted_C52_only = MAP_list[np.nanargmin(MAP_list)]
        self.params_opted_map_C52_only = param_list[np.nanargmin(MAP_list)]

    def MAP_model3_C53_only(self,params):
                
        unpacked_vector = self.unpack_vector(params)
        loss_prior = self.prior.log_prob(unpacked_vector)
        
        model3 = self.model3_ssm(params,self.X,self.Y,self.T_sub_timestamp,self.food_item_index,C_53_only=True)
        loss_time_series = model3.log_prob(self.Y,mask=self.mask)
    
        return -loss_time_series -loss_prior  
    
    @tf.function(jit_compile=True)
    def get_loss_and_grads_map_C53_only(self,params):
        with tf.GradientTape() as tape:
            tape.watch(params)
            loss = self.MAP_model3_C53_only(params)
        grads = tape.gradient(loss,params)
        return loss, grads
    
    def find_MAP_C53_only(self,loss_and_grads,N_starts=500,seed=123,max_iterations=1000,max_line_search_iterations=500,initial_inverse_hessian_scale = 1e-4,print_on=True):
        tf.random.set_seed(seed)
        MAP_list = []
        param_list = []
        for i in range(N_starts):
            if print_on:
                print(i)
            
            start = self.sample_prior()
    
            optim_results = tfp.optimizer.bfgs_minimize(
              loss_and_grads,
              initial_position=start,
              max_iterations=max_iterations,
              max_line_search_iterations=max_line_search_iterations,
              initial_inverse_hessian_estimate = initial_inverse_hessian_scale*tf.eye(len(start))
              )
    
            MAP_list.append(optim_results.objective_value)
            param_list.append(optim_results.position)
        
        self.MAP_opted_C53_only = MAP_list[np.nanargmin(MAP_list)]
        self.params_opted_map_C53_only = param_list[np.nanargmin(MAP_list)]
    