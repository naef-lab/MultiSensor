#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 03:08:53 2022

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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#%%
class Model2Actiheart(MSSModel):
    """ Class that implements Model2, which models physical activity,
        heart rate (HR) and heart rate variability (HRV). 
        Model2 is based on a 3-D system of
        stochastic differential equations, which is implemented using the
        TensorFlow Probability distribution LinearGaussianStateSpaceModel. 
        
        :param DatasetModel2Actiheart: a DatasetModel2Actiheart class
            that contains data from the Actiheart sensor
    """
    def __init__(
            self,
            DatasetModel2Actiheart,
        ) -> None:
                
                self.__dict__ = DatasetModel2Actiheart.__dict__
                self.epsilon = 1e-3
                self.w = tf.constant(2*np.pi/24)
    
                super().__init__()
                
    def sample_prior(self):
        """ Draws a sample from the model prior, returned as a TensorFlow array
        
        :returns: a real Tensor 
        :rtype: tf.float32
        """ 
        prior_sample = tfd.JointDistributionSequentialAutoBatched([
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),bijector=tfb.Log(), name='C_0_act'),
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),bijector=tfb.Log(), name='C_1_act'),
                         tfd.VonMises(loc=[0.0], concentration=[0.0], name='phi_act'),                           
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.01]),bijector=tfp.bijectors.Chain([tfb.Log(),tfb.Reciprocal()]), name='C_11'),   
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),bijector=tfb.Log(), name='C_0_bpm'),
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),bijector=tfb.Log(), name='C_1_bpm'),
                         tfd.VonMises(loc=[0.0], concentration=[0.0], name='phi_bpm'),                          
                         tfp.distributions.Normal(loc = [0.0],scale = [5.], name='C_21'),                       
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.01]),bijector=tfp.bijectors.Chain([tfb.Log(),tfb.Reciprocal()]), name='C_22'), 
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),bijector=tfb.Log(), name='C_0_hrv'),
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),bijector=tfb.Log(), name='C_1_hrv'), 
                         tfd.VonMises(loc=[0.0], concentration=[0.0], name='phi_hrv'),                         
                         tfp.distributions.Normal(loc = [0.0],scale = [5.], name='C_31'),                       
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.01]),bijector=tfp.bijectors.Chain([tfb.Log(),tfb.Reciprocal()]), name='C_33'), 
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[5.,5.,5.]),bijector=tfb.Log(), name='sigma_params'),
                         tfd.TransformedDistribution(distribution=tfp.distributions.CholeskyLKJ(dimension=2, concentration=2.),bijector=tfb.Invert(tfb.CorrelationCholesky()), name='Cholesky_flat'),
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.,1.,1.]),bijector=tfb.Log(), name='sigma_noise'),               
                         ]).sample()
        sample_vector = tf.concat(prior_sample,0)
        return sample_vector 
            
    def transform_params(self,params):
        """ Transforms an unconstrained TensorFlow array into named variables
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
  
    def unpack_vector(self,params):
        """ Converts from an unconstrained TensorFlow array to a list, 
            which is compatible with the TensorFlow Probability 
            distribution JointDistributionSequentialAutoBatched
            
        :param params: an unconstrained TensorFlow array of parameter values
        :return: list of parameter variables
        """
        params_unpacked = [params[0,tf.newaxis]
                           ,params[1,tf.newaxis]
                           ,params[2,tf.newaxis]
                           ,params[3,tf.newaxis]
                           ,params[4,tf.newaxis]
                           ,params[5,tf.newaxis]
                           ,params[6,tf.newaxis]
                           ,params[7,tf.newaxis]
                           ,params[8,tf.newaxis]
                           ,params[9,tf.newaxis]
                           ,params[10,tf.newaxis]
                           ,params[11,tf.newaxis]
                           ,params[12,tf.newaxis]
                           ,params[13,tf.newaxis]
                           ,params[14:17]
                           ,params[17,tf.newaxis]
                           ,params[18:21]]
        return params_unpacked
    
    def get_transition_probs(self,params,X):
        """ Using the parameters of the model and time of day, this function 
            returns all of the elements required to construct the transition
            model of the state space model
            
        :param params: an unconstrained TensorFlow array of parameter values
        :param X: the time vector
        :return: the mean function of each variable and all matrices required
            for the transition model for Model2
        """  
        
        delta_t = X[1]-X[0]
        C_0_act, C_1_act, phi_act, C_11, C_0_bpm, C_1_bpm, phi_bpm, C_21, C_22, C_31, C_33, C_0_hrv, C_1_hrv, phi_hrv, C_31, C_33, D_11, D_22, D_23, D_33, sigma_act, sigma_bpm, sigma_hrv = self.transform_params(params)

        mean_act = C_0_act + C_1_act*(1+tf.cos(self.w*X-phi_act))/2
        mean_bpm = C_0_bpm + C_1_bpm*(1+tf.cos(self.w*X-phi_bpm))/2 
        mean_hrv = C_0_hrv + C_1_hrv*(1+tf.cos(self.w*X-phi_hrv))/2
         
        expm_M = tf.concat([tf.stack([tf.exp(-C_11*delta_t),[0.],[0.]], 1),tf.stack([ -(C_21*tf.exp(-C_11*delta_t) - C_21*tf.exp(-C_22*delta_t))/(C_11 - C_22),tf.exp(-C_22*delta_t),[0.]], 1),tf.stack([ -(C_31*tf.exp(-C_11*delta_t) - C_31*tf.exp(-C_33*delta_t))/(C_11 - C_33),[0.], tf.exp(-C_33*delta_t)], 1)],0)
    
        P = tf.concat([tf.stack([ D_11/(2*C_11), (C_21*D_11)/(2*C_11*(C_11 + C_22)), (C_31*D_11)/(2*C_11*(C_11 + C_33))], 1),tf.stack([ (C_21*D_11)/(2*C_11*(C_11 + C_22)), (D_22*C_11**2 + C_22*D_22*C_11 + D_11*C_21**2)/(2*C_11*C_22*(C_11 + C_22)), (2*C_11**3*D_23 + 2*C_11**2*C_22*D_23 + 2*C_11**2*C_33*D_23 + 2*C_11*C_21*C_31*D_11 + C_21*C_22*C_31*D_11 + 2*C_11*C_22*C_33*D_23 + C_21*C_31*C_33*D_11)/(2*C_11*(C_11 + C_22)*(C_11 + C_33)*(C_22 + C_33))], 1),tf.stack([ (C_31*D_11)/(2*C_11*(C_11 + C_33)), (2*C_11**3*D_23 + 2*C_11**2*C_22*D_23 + 2*C_11**2*C_33*D_23 + 2*C_11*C_21*C_31*D_11 + C_21*C_22*C_31*D_11 + 2*C_11*C_22*C_33*D_23 + C_21*C_31*C_33*D_11)/(2*C_11*(C_11 + C_22)*(C_11 + C_33)*(C_22 + C_33)), (D_33*C_11**2 + C_33*D_33*C_11 + D_11*C_31**2)/(2*C_11*C_33*(C_11 + C_33))], 1)],0)
        transition_cov = P - tf.linalg.matmul(expm_M,tf.linalg.matmul(P,tf.transpose(expm_M))) + tf.linalg.diag([self.epsilon,self.epsilon,self.epsilon])  
    
        transition_matrix = expm_M
        
        initial_loc = tf.stack([0.,0.,0.],0) 
        initial_scale_diag = tf.concat([D_11/(2*C_11),(D_22*C_11**2 + C_22*D_22*C_11 + D_11*C_21**2)/(2*C_11*C_22*(C_11 + C_22)),(D_33*C_11**2 + C_33*D_33*C_11 + D_11*C_31**2)/(2*C_11*C_33*(C_11 + C_33))],0)
        initial_scale_diag = tf.sqrt(initial_scale_diag)
        
        return mean_act, mean_bpm, mean_hrv, transition_matrix, transition_cov, initial_loc, initial_scale_diag
        
    def model2(self,params,X,Y):
        """ Creates Model2 as a TensorFlow Probability 
        LinearGaussianStateSpaceModel distribution. The log liklihood
        of this model is subsequently used for parameter inference
            
        :param params: an unconstrained TensorFlow array of parameter values
        :param X: the time vector (in absolute hours)
        :return: Model2 as a tfp LinearGaussianStateSpaceModel object   
        """ 
        C_0_act, C_1_act, phi_act, C_11, C_0_bpm, C_1_bpm, phi_bpm, C_21, C_22, C_31, C_33, C_0_hrv, C_1_hrv, phi_hrv, C_31, C_33, D_11, D_22, D_23, D_33, sigma_act, sigma_bpm, sigma_hrv = self.transform_params(params)
        mean_act, mean_bpm, mean_hrv, transition_matrix, transition_cov, initial_loc, initial_scale_diag = self.get_transition_probs(params,X)

        meanf = tf.stack([mean_act,mean_bpm,mean_hrv], 0)
        
        transition_noise=tfd.MultivariateNormalTriL(
            scale_tril=tf.linalg.cholesky(transition_cov))
        
        observation_matrix = tf.stack([tf.stack([1., 0., 0.], 0),tf.stack([0., 1., 0.], 0),tf.stack([0., 0., 1.], 0)],1)

        def observation_noise(t):
            loc = meanf[:,t]
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
        
        return model  
     
    def prior_log_prob(self,params):
        """ The log probability of the prior distribution
        
        :param params: an unconstrained TensorFlow array of parameter values
        :return: the prior log probability
        """
        loss_prior = tfd.JointDistributionSequentialAutoBatched([
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),bijector=tfb.Log(), name='C_0_act'),
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),bijector=tfb.Log(), name='C_1_act'),  
                         tfd.VonMises(loc=[0.0], concentration=[0.0], name='phi_act'),                         
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.01]),bijector=tfp.bijectors.Chain([tfb.Log(),tfb.Reciprocal()]), name='C_11'),   
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),bijector=tfb.Log(), name='C_0_bpm'),
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),bijector=tfb.Log(), name='C_1_bpm'), 
                         tfd.VonMises(loc=[0.0], concentration=[0.0], name='phi_bpm'),                          
                         tfp.distributions.Normal(loc = [0.0],scale = [5.], name='C_21'),                    
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.01]),bijector=tfp.bijectors.Chain([tfb.Log(),tfb.Reciprocal()]), name='C_22'), 
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),bijector=tfb.Log(), name='C_0_hrv'),
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.]),bijector=tfb.Log(), name='C_1_hrv'), 
                         tfd.VonMises(loc=[0.0], concentration=[0.0], name='phi_hrv'),                         
                         tfp.distributions.Normal(loc = [0.0],scale = [5.], name='C_31'),                       
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[0.01]),bijector=tfp.bijectors.Chain([tfb.Log(),tfb.Reciprocal()]), name='C_33'), 
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[5.,5.,5.]),bijector=tfb.Log(), name='sigma_params'),
                         tfd.TransformedDistribution(distribution=tfp.distributions.CholeskyLKJ(dimension=2, concentration=2.),bijector=tfb.Invert(tfb.CorrelationCholesky()), name='Cholesky_flat'),          
                         tfd.TransformedDistribution(distribution=tfd.HalfNormal(scale=[1.,1.,1.]),bijector=tfb.Log(), name='sigma_noise'),                       
                         ]).log_prob(params)   
        return loss_prior 
    
    def MAP_model2_ssm(self,params):
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
        loss_prior = self.prior_log_prob(unpacked_vector)
        
        model2 = self.model2(params,self.X,self.Y)
        loss_time_series1 = model2.log_prob(self.Y,mask=self.mask)
        
        return loss_time_series1 + loss_prior  
        
    @tf.function(jit_compile=True)
    def lik_model2_ssm(self,params):
        """ The log probability density of Model2
        
        :param params: an unconstrained TensorFlow array of parameter values
        :return: the log probability density of Model1
        """
        model2 = self.model2(params,self.X,self.Y)
        loss_time_series1 = model2.log_prob(self.Y,mask=self.mask)
        
        return -loss_time_series1   
    
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
            loss = self.MAP_model2_ssm(params)
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
                    
        C_0_act = tf.math.exp(posterior[:,0])
        C_1_act = tf.math.exp(posterior[:,1])
        phi_act = posterior[:,2]     
        C_11 = tf.math.exp(posterior[:,3])
        
        C_0_bpm = tf.math.exp(posterior[:,4]) 
        C_1_bpm = tf.math.exp(posterior[:,5])
        phi_bpm = posterior[:,6]     
        C_21 = tf.math.exp(posterior[:,7]) 
        C_22 = tf.math.exp(posterior[:,8]) 
        
        C_0_hrv = tf.math.exp(posterior[:,9]) 
        C_1_hrv = tf.math.exp(posterior[:,10])
        phi_hrv = posterior[:,11]    
        C_31 = tf.math.exp(posterior[:,12]) 
        C_33 = tf.math.exp(posterior[:,13]) 
        
        D_11 = tf.math.exp(posterior[:,14])**2
        
        D_22_list = []
        D_23_list = []
        D_33_list = []
        rho_list = []
        
        for sample in range(tf.shape(posterior)[0]):
            params = posterior[sample,:]
            sigma_matrix = tf.linalg.diag(tf.math.exp(params[15:17]))
            cholesk = tfb.CorrelationCholesky().forward(params[17,tf.newaxis])
            corr = tfb.CholeskyOuterProduct().forward(cholesk)+tf.linalg.diag([0.001])
            cov = tf.matmul(sigma_matrix, tf.matmul(corr, sigma_matrix))
            D_22 = cov[0,0,tf.newaxis]
            D_23 = cov[0,1,tf.newaxis]
            D_33 = cov[1,1,tf.newaxis]
            rho = D_23/(tf.sqrt(D_22)*tf.sqrt(D_33))
            D_22_list.append(D_22)
            D_23_list.append(D_23)
            D_33_list.append(D_33)
            rho_list.append(rho)
            
        D_22 = np.array(D_22_list)
        D_23 = np.array(D_23_list)
        D_33 = np.array(D_33_list)
        rho = np.array(rho_list)
        
        sigma_act = tf.math.exp(posterior[:,18])
        sigma_bpm = tf.math.exp(posterior[:,19])
        sigma_hrv = tf.math.exp(posterior[:,20])    
        
        posterior_params_dict = {}
        variable_names = ['C_0_act','C_1_act','phi_act','C_11','C_0_bpm',
                          'C_1_bpm','phi_bpm','C_21','C_22','C_0_hrv','C_1_hrv',
                          'phi_hrv','C_31','C_33','D_11','D_22','D_23','D_33',
                          'rho','sigma_act','sigma_bpm','sigma_hrv']  
        for variable in variable_names:
            posterior_params_dict[variable] = eval(variable)
            
        self.posterior_params_dict = posterior_params_dict
   
    @tf.function(jit_compile=True)
    def predict_using_activity(self,params,X,Y,mask):
        """ Uses a Kalman forward filter to predict activity, HR and HRV
            using only activity data
            
        :param params: an unconstrained TensorFlow array of parameter values
        :param X: the time vector (in absolute hours)
        :param Y: the activity data
        :return: the filtered variables   
        """ 
        
        C_0_act, C_1_act, phi_act, C_11, C_0_bpm, C_1_bpm, phi_bpm, C_21, C_22, C_31, C_33, C_0_hrv, C_1_hrv, phi_hrv, C_31, C_33, D_11, D_22, D_23, D_33, sigma_act, sigma_bpm, sigma_hrv = self.transform_params(params)
        mean_act, mean_bpm, mean_hrv, transition_matrix, transition_cov, initial_loc, initial_scale_diag = self.get_transition_probs(params,X)

        meanf = tf.stack([mean_act,mean_bpm,mean_hrv], 0)
        
        transition_noise=tfd.MultivariateNormalTriL(
            scale_tril=tf.linalg.cholesky(transition_cov))
        
        meanf = tf.stack([mean_act,mean_bpm,mean_hrv], 0)
                
        observation_matrix = tf.stack([tf.stack([1., 0., 0.], 0)],0)
        
        def observation_noise(t):
            loc = meanf[0,t]
            scale_diag=tf.concat([sigma_act**2], 0)
            return tfd.MultivariateNormalLinearOperator(
                    loc=loc,
                    scale=tf.linalg.LinearOperatorDiag(scale_diag)) 
            
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

        return tf.transpose(meanf), filtered_means 
  
    def model_fit_metrics_hr(self):
        """ Starting from a list of posterior parameter samples, this return
            a dictionary of model fit metrics, including variance explained
            for predicting HR
            
        :param samples_list: a list of samples from the posterior parameter
            distribution
        :return: a dictionary containing model fit summary metrics
        """
    
        posterior = tf.concat(self.samples_list,0) 
        
        expl_var_circ_list = []
        expl_var_circ_and_act_list = []
        
        for sample in range(tf.shape(posterior)[0]):
            params = posterior[sample,:]
            expl_var_circ, expl_var_circ_and_act = self.get_model_fit_hr(params)
            expl_var_circ_list.append(expl_var_circ)
            expl_var_circ_and_act_list.append(expl_var_circ_and_act)
        
        model_fit_dict = {'expl_var_circ':np.array(expl_var_circ_list),
                         'expl_var_circ_and_act':np.array(expl_var_circ_and_act_list)}
        
        self.model_fit_dict_hr = model_fit_dict

        
    def get_model_fit_hr(self, params):
        """Calculates the model fit summary metrics from a single
            TensorFlow array of parameter values for HR
        
        :param params: a TensorFlow array of parameter values
        :return: summary metrics of model fit
        """
        
        meanf, filtered = self.predict_using_activity(params,self.X,self.Y[:,0,None],self.mask)
                
        Y_pred = meanf + filtered
    
        var = 1
        y_dat = self.Y.numpy()[~self.mask,1]*self.normalisation_constant[1]
        prediction_bpm = Y_pred[:,var]*self.normalisation_constant[var]
        circ_bpm = meanf[:,var]*self.normalisation_constant[var]
        
        var_data = np.var(y_dat)
    
        residual_circ = y_dat-circ_bpm[~self.mask]
        var_residual_circ = np.var(residual_circ)
        expl_var_circ = 1-var_residual_circ/var_data
         
        residual_circ_and_act = y_dat-prediction_bpm[~self.mask]
        var_residual_circ_and_act = np.var(residual_circ_and_act)
        expl_var_circ_and_act = 1-var_residual_circ_and_act/var_data
        
        return expl_var_circ, expl_var_circ_and_act
    
    @tf.function(jit_compile=True)
    def predict_using_activity_hr(self,params,X,Y,mask):
        """ Uses a Kalman forward filter to predict activity, HR and HRV
            using activity and HR data
            
        :param params: an unconstrained TensorFlow array of parameter values
        :param X: the time vector (in absolute hours)
        :param Y: the activity and HR data
        :return: the filtered variables   
        """ 
        
        C_0_act, C_1_act, phi_act, C_11, C_0_bpm, C_1_bpm, phi_bpm, C_21, C_22, C_31, C_33, C_0_hrv, C_1_hrv, phi_hrv, C_31, C_33, D_11, D_22, D_23, D_33, sigma_act, sigma_bpm, sigma_hrv = self.transform_params(params)
        mean_act, mean_bpm, mean_hrv, transition_matrix, transition_cov, initial_loc, initial_scale_diag = self.get_transition_probs(params,X)

        meanf = tf.stack([mean_act,mean_bpm,mean_hrv], 0)
        
        transition_noise=tfd.MultivariateNormalTriL(
            scale_tril=tf.linalg.cholesky(transition_cov))
        
        meanf = tf.stack([mean_act,mean_bpm,mean_hrv], 0)
        meanf_act_hr = tf.stack([mean_act,mean_bpm], 0)
        
        observation_matrix = tf.stack([tf.stack([1., 0., 0.], 0),tf.stack([0., 1., 0.], 0)],0)
        
        def observation_noise(t):
            loc = meanf_act_hr[:,t]#meanf[:2,t]
            scale_diag=tf.concat([sigma_act**2,sigma_bpm**2], 0)
            return tfd.MultivariateNormalLinearOperator(
                    loc=loc,
                    scale=tf.linalg.LinearOperatorDiag(scale_diag))
            
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

        return tf.transpose(meanf), filtered_means 
  
    def model_fit_metrics_hrv(self):
        """ Starting from a list of posterior parameter samples, this returns
            a dictionary of model fit metrics, including variance explained
            for predicting HRV
            
        :param samples_list: a list of samples from the posterior parameter
            distribution
        :return: a dictionary containing model fit summary metrics
        """
    
        posterior = tf.concat(self.samples_list,0) 
        
        expl_var_circ_list = []
        expl_var_circ_and_act_list = []
        expl_var_circ_and_act_and_hrv_list = []
        
        for sample in range(tf.shape(posterior)[0]):
            params = posterior[sample,:]
            expl_var_circ, expl_var_circ_and_act, expl_var_circ_and_act_and_hrv = self.get_model_fit_hrv(params)
            expl_var_circ_list.append(expl_var_circ)
            expl_var_circ_and_act_list.append(expl_var_circ_and_act)
            expl_var_circ_and_act_and_hrv_list.append(expl_var_circ_and_act_and_hrv)
        
        model_fit_dict = {'expl_var_circ':np.array(expl_var_circ_list),
                          'expl_var_circ_and_act':np.array(expl_var_circ_and_act_list),
                          'expl_var_circ_and_act_and_hrv':np.array(expl_var_circ_and_act_and_hrv_list)}
        
        self.model_fit_dict_hrv = model_fit_dict

    def get_model_fit_hrv(self, params):
        """Calculates the model fit summary metrics from a single
            TensorFlow array of parameter values for HR
        
        :param params: a TensorFlow array of parameter values
        :return: summary metrics of model fit
        """
        
        meanf, filtered = self.predict_using_activity(params,self.X,self.Y[:,0,None],self.mask)
                
        Y_pred = meanf + filtered
    
        var = 2
        y_dat = self.Y.numpy()[~self.mask,var]*self.normalisation_constant[var]
        prediction_bpm = Y_pred[:,var]*self.normalisation_constant[var]
        circ_bpm = meanf[:,var]*self.normalisation_constant[var]
        
        var_data = np.var(y_dat)
    
        residual_circ = y_dat-circ_bpm[~self.mask]
        var_residual_circ = np.var(residual_circ)
        expl_var_circ = 1-var_residual_circ/var_data
         
        residual_circ_and_act = y_dat-prediction_bpm[~self.mask]
        var_residual_circ_and_act = np.var(residual_circ_and_act)
        expl_var_circ_and_act = 1-var_residual_circ_and_act/var_data
        
        meanf, filtered = self.predict_using_activity_hr(params,self.X,self.Y[:,:2],self.mask)
                
        Y_pred = meanf + filtered
        
        prediction_hrv = Y_pred[:,var]*self.normalisation_constant[var]

        residual_circ_and_act_and_hrv = y_dat-prediction_hrv[~self.mask]
        var_residual_circ_and_act_and_hrv = np.var(residual_circ_and_act_and_hrv)
        expl_var_circ_and_act_and_hrv = 1-var_residual_circ_and_act_and_hrv/var_data
 
        return expl_var_circ, expl_var_circ_and_act, expl_var_circ_and_act_and_hrv
    