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
                 tol:float=1e-4,
                 ):
        
        self.M = M
        self.grady_fun = grady_fun
        self.gradt_fun = gradt_fun
        
        self.lr_rate = lr_rate
        self.max_iter = max_iter
        self.tol = tol
        
        return
    
    def __str__(self)->str:
        
        return "Gradient Descent Optimization for diffusion t-mean"
    
    def grad(self,
             z:Array,
             q:Array,
             )->Array:
        
        t = sigmoid(q)
        gradz = jnp.mean(vmap(lambda x: self.grady_fun(x,z,t))(self.X_obs), axis=0)
        gradq = jnp.mean(vmap(lambda x: self.gradt_fun(x,z,t))(self.X_obs), axis=0)*grad(sigmoid)(q)
        
        return -jnp.hstack((gradz, gradq))
    
    def local_step(self,
                   carry:Tuple[Array,Array,Array,int], 
                   )->Array:
        
        q, z, grad, idx = carry
        
        gradz = grad[:-1]
        gradq = grad[-1]
        
        q -= self.lr_rate*gradq
        z -= self.lr_rate*gradz
        
        grad = self.grad(z,q)
        
        return (q,z,grad,idx+1)
    
    def TM_step(self,
                carry:Tuple[Array,Array,Array,int], 
                )->Array:
        
        q, z, grad, idx = carry
        
        gradz = grad[:-1]
        gradq = grad[-1]
        
        q -= self.lr_rate*gradq
        z = self.M.Exp(z, -self.lr_rate*gradz)
        
        grad = self.grad(z,q)
        
        return (q,z,grad,idx+1)
    
    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        q, z, grad, idx = carry
        
        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
        
    def __call__(self, 
                 x0:Array,
                 t0:Array,
                 X_obs:Array
                 )->Tuple[Array, Array]:
        
        q0 = jnp.log(t0/(1.-t0))
        self.X_obs = X_obs
        
        grad = self.grad(x0, q0)
        if self.M.intrinsic:
            q, z, grad, idx = lax.while_loop(self.cond_fun,
                                             self.local_step,
                                             init_val=(q0,x0,grad,0),
                                             )
        else:
            q, z, grad, idx = lax.while_loop(self.cond_fun,
                                             self.TM_step,
                                             init_val=(q0,x0,grad,0),
                                             )

        t = sigmoid(q)
            
        return t, z
