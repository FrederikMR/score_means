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

class ADAM(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 grady_fun:Callable[[Array,Array, Array], Array],
                 gradt_fun:Callable[[Array, Array, Array], Array],
                 lr_rate:float=0.01,
                 beta1:float=0.9,
                 beta2:float=0.999,
                 eps:float=1e-08,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 ):
        
        self.M = M
        self.grady_fun = grady_fun
        self.gradt_fun = gradt_fun
        
        self.lr_rate = lr_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol
        
        return
    
    def __str__(self)->str:
        
        return "ADAM Optimization for diffusion t-mean"
    
    def gt(self,
           z:Array,
           q:Array,
           )->Array:
        
        t = sigmoid(q)
        gradz = jnp.mean(vmap(lambda x: self.grady_fun(x,z,t))(self.X_obs), axis=0)
        gradq = jnp.mean(vmap(lambda x: self.gradt_fun(x,z,t))(self.X_obs), axis=0)*grad(sigmoid)(q)
        
        return -jnp.hstack((gradz, gradq))
    
    def adam_update(self,
                    mt:Array,
                    vt:Array,
                    gt:Array,
                    t:int,
                    ):
        
        mt = self.beta1*mt+(1.-self.beta1)*gt
        vt = self.beta2*vt+(1-self.beta2)*(gt**2)
        
        mt_hat = mt/(1.-(self.beta1)**(t))
        vt_hat = vt/(1.-(self.beta2)**(t))
        
        step = self.lr_rate*mt_hat/(jnp.sqrt(vt_hat)+self.eps)
        
        return mt, vt, step
    
    def local_step(self,
                   carry:Tuple[Array,Array,Array,Array,Array,int],
                   )->Array:
        
        q, z, gt, mt, vt, idx = carry
        
        mt, vt, step = self.adam_update(mt, vt, gt, idx)
        
        gradz = step[:-1]
        gradq = step[-1]
        
        q -= gradq
        z -= gradz

        gt = self.gt(z, q)
        
        return (q, z, gt, mt, vt, idx+1)
    
    def TM_step(self,
                carry:Tuple[Array,Array,Array,Array,Array,int],
                )->Array:
        
        q, z, gt, mt, vt, idx = carry
        
        mt, vt, step = self.adam_update(mt, vt, gt, idx)
        
        gradz = step[:-1]
        gradq = step[-1]
        
        q -= gradq
        z = self.M.Exp(z, -gradz)
        
        gt = self.gt(z, q)
        
        return (q, z, gt, mt, vt, idx+1)
    
    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array,Array,Array,int],
                 )->Array:
        
        q, z, gt, mt, vt, idx = carry
        
        norm_grad = jnp.linalg.norm(gt.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
        
    def __call__(self, 
                 x0:Array,
                 t0:Array,
                 X_obs:Array
                 )->Tuple[Array, Array]:
        
        q0 = jnp.log(t0/(1.-t0))
        self.X_obs = X_obs
        
        gt, mt, vt = self.gt(x0, q0), jnp.zeros(len(x0)+1), jnp.zeros(len(x0)+1)
        if self.M.intrinsic:
            q, z, gt, mt, vt, idx = lax.while_loop(self.cond_fun, 
                                                   self.local_step,
                                                   init_val=(q0, x0, gt, mt, vt, 1),
                                                   )
        else:
            q, z, gt, mt, vt, idx = lax.while_loop(self.cond_fun, 
                                                   self.TM_step,
                                                   init_val=(q0, x0, gt, mt, vt, 1),
                                                   )
        t = sigmoid(q)
            
        return t, z
