#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:37:02 2024

@author: fmry
"""

#%% Moules

from .setup import *

from score_means.manifolds import RiemannianManifold

#%% Gradient Descent

class GradientDescent(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 grady_fun:Callable[[Array,Array, Array], Array],
                 gradt_fun:Callable[[Array, Array, Array], Array],
                 lr_rate:float=0.01,
                 max_iter:int=1000,
                 ):
        
        self.M = M
        self.grady_fun = grady_fun
        self.gradt_fun = gradt_fun
        
        self.lr_rate = lr_rate
        self.max_iter = max_iter
        
        return
    
    def __str__(self)->str:
        
        return "Gradient Descent Optimization for diffusion t-mean"
    
    def local_step(self,
                   carry:Tuple[Array,Array], 
                   idx:int,
                   )->Array:
        
        q, z = carry
        
        t = sigmoid(q)
        gradz = -jnp.mean(vmap(lambda x: self.grady_fun(x,z,t))(self.X_obs), axis=0)
        gradq = -jnp.mean(vmap(lambda x: self.gradt_fun(x,z,t))(self.X_obs), axis=0)*grad(sigmoid)(q)
        
        q -= self.lr_rate*gradq
        z -= self.lr_rate*gradz
        
        return ((q,z),)*2
    
    def TM_step(self,
                carry:Tuple[Array,Array],
                idx:int,
                )->Array:
        
        q, z = carry
        
        t = sigmoid(q)
        gradz = -jnp.mean(vmap(lambda x: self.grady_fun(x,z,t))(self.X_obs), axis=0)
        gradq = -jnp.mean(vmap(lambda x: self.gradt_fun(x,z,t))(self.X_obs), axis=0)*grad(sigmoid)(q)
        
        q -= self.lr_rate*gradq
        z = self.M.Exp(z, -self.lr_rate*gradz)
        
        return ((q,z),)*2
        
    def __call__(self, 
                 x0:Array,
                 t0:Array,
                 X_obs:Array
                 )->Tuple[Array, Array]:
        
        self.X_obs = X_obs
        
        q0 = 1./t0-1.
        
        if self.M.intrinsic:
            out, _ = lax.scan(self.local_step,
                              init=(q0,x0),
                              xs=jnp.ones(self.max_iter))
        else:
            out, _ = lax.scan(self.TM_step,
                              init=(q0,x0),
                              xs=jnp.ones(self.max_iter))
            
            
        q, z = out[0], out[1]
        t = sigmoid(q)
            
        return t, z
