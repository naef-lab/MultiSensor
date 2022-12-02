#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:12:17 2022

@author: phillips
"""

import os
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

class MSSModel:
    """
    This is the super class of Model1, Model2 and Model3. It contains methods
    for parameter optimisation and parameter sampling with MCMC that are used
    by all models. Not to be instantiated.
    """
         
    def find_MAP(self,
                 loss_and_grads,
                 N_starts=500,
                 seed=123,
                 max_iterations=1000,
                 max_line_search_iterations=500,
                 initial_inverse_hessian_scale = 1e-4,
                 print_on=True):
        """ Function to find the MAP parameter point estimate of the model
            using the BFGS algorithm.
            Uses multiple restarts using samples from the prior distribution as
            initial starting points.
            
        :param loss_and_grads: callable that accepts a point as a real Tensor 
            and returns a tuple of Tensors containing the value 
            of the MAP function and its gradient at that point
        :param N_starts: the number of random initialisations for optimisation
        :param seed: the seed of the optimiser
        :param max_iterations: The maximum number of iterations for BFGS updates.
        :param max_line_search_iterations: The maximum number of iterations 
            for the line search algorithm.
        :param initial_inverse_hessian_scale: the starting estimate for the 
            inverse of the Hessian at the initial point
        :param print_on: prints the current iteration number in terms of the 
            number of initialisations from prior samples  

        """
        tf.random.set_seed(seed)
        MAP_list, param_list = [[],[]]
        
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
        
        self.MAP_list = MAP_list
        self.param_list = param_list
        self.MAP_opted = MAP_list[np.nanargmin(MAP_list)]
        self.params_opted_map = param_list[np.nanargmin(MAP_list)]
        
    @tf.function(jit_compile=True)
    def run_chain(self,
                  initial_state, 
                  num_results, 
                  num_burnin_steps, 
                  kernel, 
                  seed = [1,1]):
        """ Implements Markov chain Monte Carlo (MCMC), which is used to
            sample the posterior parameter distribution.
        
        :param initial_state: the current position of the Markov chain
        :param num_results: number of Markov chain samples
        :param num_burnin_steps: number of chain steps to take before starting 
            to collect results
        :param kernel: An instance of tfp.mcmc.TransitionKernel that
            implements one step of the Markov chain
        :param seed: seed for sampler
        """
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=kernel,
            seed = seed,
            trace_fn=lambda current_state, kernel_results: kernel_results)
      
    def first_MCMC_run(self, 
                       initial_state, 
                       step_size = 3e-3, 
                       num_burnin_steps = 10000, 
                       num_results = 10000):
        """ Inital an initial MCMC run to estimate the standard deviation of 
            the posterior parameter distribution. The standard deviation is 
            then used to tune the step size of the full MCMC run.
            
        :param initial_state: the current position of the Markov chain
        :param step_size: the step size for the leapfrog integrator
        :param num_burnin_steps: number of chain steps to take before starting 
            to collect results
        :param num_results: number of Markov chain samples
        """
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.unnormalized_posterior,
            num_leapfrog_steps=5,
            step_size=step_size)
        
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8))
    
        samples, kernel_results = self.run_chain(initial_state=initial_state, 
                                                 num_results=num_results, 
                                                 num_burnin_steps=num_burnin_steps, 
                                                 kernel=kernel)
        
        approx_posterior_std = np.std(samples,axis=0)
        approx_posterior_std = approx_posterior_std/np.max(approx_posterior_std)
        self.step_size_optimised = approx_posterior_std*1e-1
        
    def sample_posterior(self, 
                         initial_state, 
                         step_size, 
                         num_chains = 4, 
                         num_burnin_steps = 10000, 
                         num_results = 10000):
        """ Full HMC to sample from the posterior parameter distribution.
            The MCMC samples are saved as a list, which is then used for
            the main results. MCMC diagnostics are also saved, including the
            acceptance_rate, the potential scale reduction (rhat)
            and the log probability
            
        :param initial_state: the current position of the Markov chain
        :param step_size: the step size for the leapfrog integrator
        :param num_chains: the number of independent MCMC chains
        :param num_burnin_steps: number of chain steps to take before starting 
            to collect results
        :param num_results: number of Markov chain samples
        
        """
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.unnormalized_posterior,
            num_leapfrog_steps=5,
            step_size=step_size)
        
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8))
    
        samples_list = []
        kernel_results_list = []
        
        for i in range(num_chains):
            print(i)
            samples, kernel_results = self.run_chain(initial_state=initial_state, 
                                                     num_results=num_results, 
                                                     num_burnin_steps=num_burnin_steps, 
                                                     kernel=kernel, 
                                                     seed = [i,i])
            samples_list.append(samples)
            kernel_results_list.append(kernel_results)
        
        acceptance_rate = [kernel_results_list[chain].inner_results.is_accepted.numpy().mean() for chain in range(len(kernel_results_list))]
    
        chain_states = tf.stack(samples_list,1)
        rhat = tfp.mcmc.potential_scale_reduction(chain_states, 
                                                  independent_chain_ndims=1)  
        
        target_log_prob = [kernel_results_list[chain].inner_results.accepted_results.target_log_prob for chain in range(len(kernel_results_list))] 
        
        mcmc_diagnostics = {'acceptance_rate': acceptance_rate, 
                            'rhat' : rhat, 
                            'target_log_prob' : target_log_prob}
        
        self.samples_list = samples_list
        self.mcmc_diagnostics = mcmc_diagnostics
        
        