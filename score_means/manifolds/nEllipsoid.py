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

class nEllipsoid(RiemannianManifold):
    def __init__(self,
                 dim:int=2,
                 params:Array=None,
                 )->None:
        
        if params == None:
            params = jnp.ones(dim+1, dtype=jnp.float32)
        self.params = params
        
        self.dim = dim
        self.emb_dim = dim +1
        super().__init__(f=self.f_stereographic, 
                         invf = self.invf_stereographic,
                         intrinsic=False)
        
        return
    
    def __str__(self)->str:
        
        return f"Sphere of dimension {self.dim} embedded in R^{self.emb_dim}"
    
    def f_stereographic(self,
                        z:Array,
                        )->Array:
        
        s2 = jnp.sum(z**2)
        
        return self.params*jnp.hstack(((1-s2), 2*z))/(1+s2)

    def invf_stereographic(self, 
                           x:Array,
                           )->Array:
        
        x /= self.params
        
        x0 = x[0]
        return x[1:]/(1+x0)
    
    def M_proj(self,
               x:Array,
               )->Array:
        
        x /= self.params
        
        return self.params*x/jnp.linalg.norm(x)
    
    def TM_proj(self,
               x:Array,
               v:Array,
               )->Array:
        
        x /= self.params
        v /= self.params
        
        return self.params*(v-jnp.dot(v,x)*x)
    
    def dist(self,
             x:Array,
             y:Array
             )->Array:
        
        x /= self.params
        y /= self.params
        
        return jnp.arccos(jnp.dot(x,y))
    
    def Exp(self,
            x:Array,
            v:Array,
            T:float=1.0,
            )->Array:
        
        x /= self.params
        v /= self.params
        
        norm = jnp.linalg.norm(v)
        
        return (jnp.cos(norm*T)*x+jnp.sin(norm*T)*v/norm)*self.params
    
    def Log(self,
            x:Array,
            y:Array
            )->Array:
        
        x /= self.params
        y /= self.params
        
        dot = jnp.dot(x,y)
        dist = jnp.arccos(jnp.dot(x,y))
        val = y-dot*x
        
        return self.params*dist*val/jnp.linalg.norm(val)
    
    def Geodesic(self,
                 x:Array,
                 y:Array,
                 t_grid:Array=None,
                 )->Array:
        
        if t_grid is None:
            t_grid = jnp.linspace(0.,1.,99, endpoint=False)[1:]
        
        x_s = x/self.params
        
        v = self.Log(x,y)
        
        gamma = self.params*vmap(lambda t: self.Exp(x_s, v,t))(t_grid)
        
        return jnp.vstack((x,gamma,y))