#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:34:16 2024

@author: fmry
"""

#%% Modules

from .setup import *

from .nEllipsoid import nEllipsoid

#%% nSphere Manifold

class nSphere(nEllipsoid):
    def __init__(self,
                 dim:int=2,
                 )->None:
        super().__init__(dim=dim, params=jnp.ones(dim+1, dtype=jnp.float32))
        
        return
    
    def __str__(self)->str:
        
        return f"Sphere of dimension {self.dim} in {self.coordinates} coordinates equipped with the pull back metric"
    
    def heat_kernel(self, 
                    x:Array,
                    y:Array,
                    t:float, 
                    N_terms:int=100,
                    )->float:
        
        def sum_term(l:int, C_l:float) -> float:
        
            return jnp.exp(-0.5*l*(l+m1)*t)*(2*l+m1)*C_l
        
        def update_cl(l:int, Cl1:float, Cl2:float) -> float:
            
            return (2*(l-1+alpha)*xy_dot*Cl1-(l+2*alpha-2)*Cl2)/l
        
        def step(carry:Tuple[float,float,float], l:int) -> Tuple[Tuple[float, float, float], None]:

            val, Cl1, Cl2 = carry

            C_l = update_cl(l, Cl1, Cl2)

            val += sum_term(l, C_l)

            return (val, C_l, Cl1), None

        x1 = x
        y1 = y
        xy_dot = jnp.dot(x1,y1)
        m1 = self.dim-1
        Am_inv = jnp.exp(jscipy.special.gammaln((self.dim+1)*0.5))/(2*jnp.pi**((self.dim+1)*0.5))
        
        alpha = m1*0.5
        C_0 = 1.0
        C_1 = 2*alpha*xy_dot
        
        val = sum_term(0, C_0) + sum_term(1, C_1)
        
        grid = jnp.arange(2,N_terms,1)
        
        val, _ = lax.scan(step, (val, C_1, C_0), xs=grid)
            
        return val[0]*Am_inv/m1