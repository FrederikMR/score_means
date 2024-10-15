#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:48:35 2024

@author: fmry
"""

#%% Modules

from .setup import *

from score_means.manifolds import RiemannianManifold

#%% Local Sampling

class RiemannianSDE(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 drift_fun:Callable,
                 diffusion_fun:Callable,
                 dt_steps:int=1000,
                 seed:int=2712,
                 )->None:
        
        self.M = M
        self.dt_steps = dt_steps
        self.seed = seed
        self.key = jrandom.key(seed)
        
        self.drift_fun = drift_fun
        self.diffusion_fun = diffusion_fun
        
        self.dtype = None
        
        if M.intrinsic:
            self.dim = M.dim
            self.sample = self.local_sampling
        else:
            self.dim = M.emb_dim
            self.sample = self.TM_sampling
        
        return
    
    def W_sample(self,
                  dt:Array,
                  n_samples:int=1,
                  )->Array:
        
        key, subkey = jrandom.split(self.key)
        self.key = key
        
        z = jrandom.normal(subkey, shape=(self.dt_steps, n_samples, self.dim))
        
        return z.squeeze()

    def local_sampling(self, 
                       z:Array, 
                       step:Tuple[Array,Array,Array],
                       )->Tuple[Array,Array]:
        
        dt, t, dW = step
        t += dt

        if z.ndim == 1:
            drift_term = self.drift_fun(t,z)
            diffusion_term = self.diffusion_fun(t,z)
        else:
            drift_term = vmap(self.drift_fun, in_axes=(None,0))(t,z)
            diffusion_term = vmap(self.diffusion_fun, in_axes=(None,0))(t,z)
            
        det = (drift_term*dt.reshape(-1,1)).squeeze()
        stoch = jnp.einsum('...ik,...k->...i', diffusion_term, dW)
        
        z += det+stoch
        z = z.astype(self.dtype)
        
        return (z,)*2
    
    def TM_sampling(self,
                    z:Array, 
                    step:Tuple[Array,Array,Array],
                    )->Tuple[Array,Array]:
        
        dt, t, dW = step
        
        if z.ndim == 1:
            dW = self.M.TM_proj(z,dW)
            drift_term = self.drift_fun(t,z)
            diffusion_term = self.diffusion_fun(t,z)
        else:
            dW = vmap(self.M.TM_proj)(z,dW)
            drift_term = vmap(self.drift_fun, in_axes=(None,0))(t,z)
            diffusion_term = vmap(self.diffusion_fun, in_axes=(None,0))(t,z)
            
        det = (drift_term*dt.reshape(-1,1)).squeeze()
        stoch = jnp.einsum('...ik,...k->...i', diffusion_term, dW)
        
        z = vmap(self.M.Exp)(z,det+stoch)
        
        return (z,)*2
    
    def __call__(self,
                 z0:Array,
                 T:Array=1.0,
                 )->Tuple[Array,Array,Array,Array]:
        
        self.dtype = z0.dtype
        
        if z0.ndim == 1:
            n_samples = 1
        else:
            n_samples = len(z0)
            
        dt = jnp.array([T/self.dt_steps]*self.dt_steps)
        t = jnp.cumsum(dt)
        dW = jnp.einsum('i,i...j->i...j', 
                        jnp.sqrt(dt), 
                        self.W_sample(dt, n_samples=n_samples)
                        )

        _, xt = lax.scan(self.sample,
                         init = z0,
                         xs = (dt,t,dW),
                         )
        
        return dt, t, dW, xt