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

class Cylinder(RiemannianManifold):
    def __init__(self,
                 r:float=1.0,
                 seed:int=2712,
                 )->None:

        self.r = r
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
        
        theta = z[0]
        v = z[1]
        
        return jnp.hstack((self.r*jnp.cos(theta), self.r*jnp.sin(theta), v))
    
    def invf_standard(self,
                      x:Array,
                      )->Array:
        
        thetea = jnp.arctan2(x[1], x[0])
        v = x[2]
        
        return jnp.hstack((theta, v))
        