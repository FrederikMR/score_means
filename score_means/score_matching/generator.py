#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:32:33 2024

@author: fmry
"""

#%% Modules

from .setup import *

from score_means.sampling import BrownianMotion
from score_means.manifolds import RiemannianManifold

#%% Generator

class BrownianSampler(object):
    def __init__(self,
                 M:RiemannianManifold,
                 x0:Array,
                 sigma:float=1.0,
                 x0_samples:int=32,
                 xt_samples:int=16,
                 dt_steps:int=100,
                 eps:float=1e-2,
                 T:float=1.0,
                 T_sample:bool = False,
                 t0:float = 0.1,
                 seed:int=2712,
                 )->None:
        
        self.M = M
        self.sampler = BrownianMotion(M, dt_steps, seed)
        
        self.x0 = x0
        self.sigma = sigma
        
        self.x0_samples = x0_samples
        self.xt_samples = xt_samples
        self.N_samples = x0_samples*xt_samples
        self.dt_steps = dt_steps
        
        self.eps = eps
        self.T = T
        
        self.T_sample = T_sample
        self.t0 = t0
            
        if M.intrinsic:
            self.dim = M.dim
        else:
            self.dim = M.emb_dim
            
        self.seed = seed
        self.key = jrandom.key(seed)
            
        return
        
    def __str__(self)->str:
        
        return "Generator for Samples of Brownian Motion on Manifold"
    
    def t_sample(self,
                 )->Array:
        
        key, subkey = jrandom.split(self.key)
        self.key = key
        
        t = jrandom.uniform(subkey, shape=(1,), minval=self.eps, maxval=self.T)
        
        return z.squeeze()
        
    def __call__(self)->Tuple[Array, Array, Array, Array, Array]:
        
        while True:
            
            x0s = self.M.sample(self.x0_samples, x0=self.x0, sigma=self.sigma).squeeze()
            if not self.T_sample:
                t = self.t0
            else:
                t = self.t_sample().squeeze() 
                
            x0s = jnp.tile(x0s, (self.xt_samples, 1, 1)).reshape(-1,self.dim)
            dt, t, dW, xt = self.sampler(x0s, t)
            
            x0s = jnp.tile(x0s, (self.dt_steps, 1, 1))
            t = jnp.tile(t,(1, self.N_samples)).reshape(self.dt_steps, self.N_samples, 1)
            dt = jnp.tile(dt,(1, self.N_samples)).reshape(self.dt_steps, self.N_samples, 1)

            yield jnp.concatenate((x0s, xt, t, dW, dt), axis=-1)
