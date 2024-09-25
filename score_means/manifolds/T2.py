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

class T2(RiemannianManifold):
    def __init__(self,
                 R:float=3.0,
                 r:float=1.0,
                 seed:int=2712,
                 )->None:
        
        self.R = R
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
        phi = z[1]
        
        cos_theta = jnp.cos(theta)
        
        x1 = (self.R+self.r*cos_theta)*jnp.cos(phi)
        x2 = (self.R+self.r*cos_theta)*jnp.sin(phi)
        x3 = self.r*jnp.sin(theta)
        
        return jnp.hstack((x1, x2, x3))
        