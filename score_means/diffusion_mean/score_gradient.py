#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:29:52 2024

@author: fmry
"""

#%% Modules

from .setup import *

from score_means.manifolds import RiemannianManifold

#%% Score Gradients

class ScoreGradient(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 s1_model:Tuple[Array, Array, Array],
                 s2_model:Tuple[Array, Array, Array]=None,
                 st_model:Tuple[Array, Array, Array]=None,
                 )->None:
        
        self.M = M
        self.s1_model = s1_model
        if s2_model is None:
            self.s2_model = jacfwd(self.s1_model, argnums=1)
        else:
            self.s2_model = s2_model
        if st_model is not None:
            self.gradt = st_model
        else:
            self.st_model = None
            
    def TM_hess(self,
                x:Array,
                v:Array,
                h:Array,
                )->Array:
    
        val1 = self.M.TM_proj(x, h)
        val2 = v-self.M.TM_proj(x, v)
        val3 = jacfwd(self.M.TM_proj, argnums=0)(x,val2)
        
        return val1+val3
    
    def grad_local(self, 
                   x:Array, 
                   v:Array,
                   )->Array:
        
        Jf = self.M.Jf(x)

        return jnp.einsum('ij,i->j', Jf, v)
    
    def hess_local(self, 
                   x:Array, 
                   v:Array, 
                   h:Array,
                   )->Array:
        
        val1 = self.M.Jf(x)
        val2 = jacfwd(lambda x1: self.M.Jf(x1))(x)
        
        term1 = jnp.einsum('jl,li,jk->ik', h, val1, val1)
        term2 = jnp.einsum('j,jik->ik', v, val2)
        
        return term1+term2
        
    def grady(self, x:Array, y:Array, t:Array)->Array:
        
        return self.M.TM_proj(y, self.s1_model(x,y,t))
        
        return self.s1_model(x,y,t)#self.M.TM_proj(x, self.s1_model(x,y,t))
    
    def gradt(self, x:Array, y:Array, t:Array)->Array:
        
        s1 = self.s1_model(x,y,t)
        s2 = self.s2_model(x,y,t)
        
        if self.M.intrinsic:
            norm_s1 = jnp.dot(s1,s1)
            laplace_beltrami = jnp.trace(s2)+.5*jnp.dot(s1,jacfwd(lambda z: jnp.linalg.slogdet(self.M.G(z))[1])(x).squeeze())
        else:
            #s2 = self.TM_hess(y, s1, s2)
            #norm_s1 = jnp.dot(s1,s1)
            #laplace_beltrami = jnp.trace(self.TM_hess(y, s1, s2))
            z = self.M.invf(y)
            s2 = self.hess_local(z, s1, s2)
            s1 = self.grad_local(z, s1)
            
            norm_s1 = jnp.dot(s1,s1)
            laplace_beltrami = jnp.trace(s2)+.5*jnp.dot(s1,jacfwd(lambda z: jnp.linalg.slogdet(self.M.G(z))[1])(z).squeeze())
    
        return 0.5*(laplace_beltrami+norm_s1)

        