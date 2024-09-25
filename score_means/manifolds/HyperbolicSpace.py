#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:38:54 2024

@author: fmry
"""

#%% Modules

from .setup import *

from .manifold import RiemannianManifold

#%% Code

class HyperbolicSpace(RiemannianManifold):
    def __init__(self,
                 dim:int=2,
                 eps:float=1e-10,
                 seed:int=2712,
                 )->None:

        self.dim = dim
        self.emb_dim = dim+1
        self.eps = eps
        
        self.seed = seed
        self.key = jrandom.key(seed)
        
        super().__init__(f=None, 
                         invf = None,
                         intrinsic=False)
        
        return
    
    def __str__(self)->str:
        
        return f"Symmetric Positive Definite Matrices of dimension {self.dim} embedded in R^{self.emb_dim}"
    
    def dot(self,
            x:Array,
            y:Array,
            )->Array:
        
        return jnp.sum(x[:-1]*y[:-1])-x[-1]*y[-1]
    
    def TM_proj(self,
               x:Array,
               v:Array,
               )->Array:
        
        return v+self.dot(x,v)*x
    
    def sample(self,
               N:int,
               x0:Array,
               sigma:float=1.0,
               )->Array:
        
        key, subkey = jrandom.split(self.key)
        self.key = key
        z = jrandom.normal(subkey, shape=(N, self.dim))+x0[:-1]
        
        res = jnp.zeros((N, self.emb_dim))
        
        res = res.at[:,:-1].set(z)
        res = res.at[:,-1].set(jnp.sqrt(jnp.sum(z**2, axis=1)+1))

        return res
    
    def dist(self,
             x:Array,
             y:Array
             )->Array:
        
        return lax.cond(jnp.linalg.norm(x-y)>self.eps,
                        lambda *_: jnp.arccosh(-self.dot(x, y)),
                        lambda *_: 0.0,
                        )
    
    def Exp(self,
            x:Array,
            v:Array,
            T:float=1.0,
            )->Array:
        
        v = T*v
        
        inner_product = jnp.sqrt(self.dot(v, v))
        
        return lax.cond(jnp.linalg.norm(v)>self.eps,
                        lambda *_: jnp.cosh(inner_product)*x+jnp.sinh(inner_product)*v/inner_product,
                        lambda *_: x,
                        )
    
    def Log(self,
            x:Array,
            y:Array
            )->Array:
        
        inner_product = self.dot(x,y)
        dist = self.dist(x,y)
        
        return lax.cond(dist > self.eps,
                        lambda *_: dist*(y-inner_product*x)/jnp.linalg.norm((y-inner_product*x)),
                        lambda *_: jnp.zeros_like(x))