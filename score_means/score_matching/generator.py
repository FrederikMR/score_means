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
                 repeats:int=32,
                 x_samples:int=1,
                 t_samples:int=10,
                 dt_steps:int=100,
                 T:float=1.0,
                 T_sample:bool = False,
                 t0:float = 0.1,
                 seed:int=2712,
                 )->None:
        
        self.M = M
        self.sampler = BrownianMotion(M, dt_steps, seed)
        
        self.repeats = repeats
        self.x_samples = x_samples
        if t_samples > dt_steps:
            self.t_samples = dt_steps
        else:
            self.t_samples = t_samples
        self.N_sim = x_samples*repeats
        
        self.T = T
        
        self.T_sample = T_sample
        self.t0 = t0
        
        if x0.ndim == 1:
            x0s = jnp.tile(x0, (repeats,1))
            self.x0s_init = x0s
            self.x0s = x0s
        else:
            self.x0s_init = x0
            self.x0s = x0
            
        if M.intrinsic:
            self.dim = M.dim
        else:
            self.dim = M.emb_dim
            
        return
        
    def __str__(self)->str:
        
        return "Generator for Samples of Brownian Motion on Manifold"
        
    def __call__(self)->Tuple[Array, Array, Array, Array, Array]:
        
        while True:
            
            dt, t, dW, xt = self.sampler(self.x0s, self.T)

            self.x0s = xt[-1] 
            
            if jnp.isnan(jnp.sum(xt)):
                self.x0s = self.x0s_init
                
            x0s = jnp.tile(jnp.repeat(self.x0s,
                                      self.x_samples,
                                      axis=0),
                           (self.sampler.dt_steps,1,1)
                           )
            t = jnp.tile(t, 
                          (self.N_sim, 1)).T.reshape(self.sampler.dt_steps, 
                                                     self.N_sim, 
                                                     1)
            dt = jnp.tile(dt, 
                          (self.N_sim, 1)).T.reshape(self.sampler.dt_steps, 
                                                     self.N_sim, 
                                                     1)
            
            if not self.T_sample:
                inds = jnp.array(random.sample(range(self.sampler.dt_steps), self.t_samples))
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
