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
                 N_samples:int=32,
                 dt_steps:int=100,
                 T:float=1.0,
                 T_sample:bool = False,
                 t0:float = 0.1,
                 seed:int=2712,
                 )->None:
        
        self.M = M
        self.sampler = BrownianMotion(M, dt_steps, seed)
        
        self.x0 = x0
        self.sigma = sigma
        
        self.N_samples = N_samples
        self.N_out = N_samples
        self.dt_steps = dt_steps
        
        self.T = T
        
        self.T_sample = T_sample
        self.t0 = t0
            
        if M.intrinsic:
            self.dim = M.dim
        else:
            self.dim = M.emb_dim
            
        return
        
    def __str__(self)->str:
        
        return "Generator for Samples of Brownian Motion on Manifold"
        
    def __call__(self)->Tuple[Array, Array, Array, Array, Array]:
        
        while True:
            
            x0s = self.M.sample(self.N_samples, x0=self.x0, sigma=self.sigma)
            dt, t, dW, xt = self.sampler(x0s, self.T)
                
            x0s = jnp.tile(x0s,
                           (self.dt_steps,1,1)
                           )
            
            t = jnp.tile(t,
                         (self.N_samples, 1)).T.reshape(self.dt_steps, 
                                                        self.N_samples, 
                                                        1)
            dt = jnp.tile(dt, 
                          (self.N_samples, 1)).T.reshape(self.dt_steps, 
                                                         self.N_samples, 
                                                         1)
                                                         
            if not self.T_sample:
                inds = jnp.array(random.sample(range(self.dt_steps), self.dt_steps))
                x0s = x0s[inds]
                xt = xt[inds]
                t = t[inds]
                dt = dt[inds]
                dW = dW[inds]
            else:
                inds = jnp.argmin(jnp.abs(t-self.t0))
                x0s = jnp.expand_dims(x0s[inds], axis=0)
                xt = jnp.expand_dims(xt[inds], axis=0)
                t = jnp.expand_dims(t[inds], axis=0)
                dt = jnp.expand_dims(dt[inds],axis=0)
                dW = jnp.expand_dims(dW[inds], axis=0)
                
            yield jnp.concatenate((x0s,
                                   xt,
                                   t,
                                   dW,
                                   dt,
                                   ), axis=-1)
