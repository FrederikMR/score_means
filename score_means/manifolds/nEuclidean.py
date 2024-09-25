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

class nEuclidean(RiemannianManifold):
    def __init__(self,
                 dim:int=2,
                 seed:int=2712,
                 )->None:

        self.dim = dim
        self.emb_dim = dim
        
        self.seed = seed
        self.key = jrandom.key(seed)
        
        super().__init__(G=self.metric,
                         f=lambda x: x, 
                         invf=lambda x: x,
                         intrinsic=True,
                         )
        
        return
    
    def __str__(self)->str:
        
        return f"Euclidean manifold of dimension {self.dim} in standard coordinates"
    
    def sample(self,
               N:int,
               x0:Array,
               sigma:float=1.0,
               )->Array:
        
        key, subkey = jrandom.split(self.key)
        self.key = key
        z = jrandom.normal(subkey, shape=(N, self.dim))
        
        return x0 + sigma*z
    
    def metric(self,
               z:Array,
               )->Array:
        
        return jnp.eye(self.dim)
    
    def dist(self,
             z1:Array,
             z2:Array
             )->Array:
        
        return jnp.linalg.norm(z2-z1)
    
    def Geodesic(self,
                 x:Array,
                 y:Array,
                 t_grid:Array=None,
                 )->Array:
        
        if t_grid is None:
            t_grid = jnp.linspace(0.,1.,100)
        
        return x+(y-x)*t_grid.reshape(-1,1)
    