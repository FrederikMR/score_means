#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

from .setup import *

####################

from .manifold import RiemannianManifold

#%% Code

class H2(RiemannianManifold):
    def __init__(self,
                 seed:int=2712,
                 )->None:

        self.dim = 2
        self.emb_dim = 3
        
        self.seed = seed
        self.key = jrandom.key(seed)
        
        super().__init__(G=None,
                         f=self.f_standard, 
                         invf=None,
                         intrinsic=True,
                         )
        
        return
    
    def __str__(self)->str:
        
        return "Hyperbolic Paraboloid equipped with the pull back metric"
    
    def sample(self,
               N:int,
               x0:Array,
               sigma:float=1.0,
               )->Array:
        
        key, subkey = jrandom.split(self.key)
        self.key = key
        z = jrandom.normal(subkey, shape=(N, self.dim))
        
        return x0 + sigma*z
    
    def f_standard(self,
                   z:Array,
                   )->Array:
        
        alpha = z[0]
        beta = z[1]
        
        x1 = jnp.cosh(alpha)
        x2 = jnp.sinh(alpha)*jnp.cos(beta)
        x3 = jnp.sinh(alpha)*jnp.sin(beta)
        
        return jnp.hstack((x1, x2, x3))
    
    def invf_standard(self,
                      x:Array,
                      )->Array:
        
        alpha = jnp.arcosh(x[0])
        beta = jnṕ.arctan2(x[2], x[1])
        
        return jnp.hstack((alpha, beta))
        