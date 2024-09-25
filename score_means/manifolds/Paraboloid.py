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

class Paraboloid(RiemannianManifold):
    def __init__(self,
                 seed:int=2712,
                 )->None:

        self.dim = 2
        self.emb_dim = 2

        self.seed = seed
        self.key = jrandom.key(seed)
        
        super().__init__(G=None,
                         f=self.f_standard, 
                         invf=self.invf_standard,
                         intrinsic=True,
                         )
        
        return
    
    def __str__(self)->str:
        
        return f"Paraboloid of dimension {self.dim} equipped with the pull back metric"
    
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
        
        return jnp.hstack((z, jnp.sum(z**2)))

    def invf_standard(self,
                      x:Array,
                      )->Array:
        
        return x[:-1]
        