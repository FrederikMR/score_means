#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:48:35 2024

@author: fmry
"""

#%% Modules

from .setup import *

from score_means.manifolds import RiemannianManifold
from .riemannian_sde import RiemannianSDE

#%% Local Sampling

class BrownianMotion(RiemannianSDE):
    def __init__(self,
                 M:RiemannianManifold,
                 dt_steps:int=1000,
                 seed:int=2712,
                 )->None:
        
        if M.intrinsic:
            drift_fun = self.local_drift
            diffusion_fun = self.local_diffusion
        else:
            drift_fun = self.TM_drift
            diffusion_fun = self.TM_diffusion
        
        
        super().__init__(M=M,
                         drift_fun=drift_fun,
                         diffusion_fun=diffusion_fun,
                         dt_steps=dt_steps,
                         seed=seed,
                         )

        return
    
    def __str__(self)->str:
        
        return "Riemannian Brownian Motion Class"
    
    def local_drift(self, 
                    t:Array,
                    z:Array, 
                    )->Array:
        
        ginv = self.M.Ginv(z)
        Chris = self.M.Chris(z)
        
        return 0.5*jnp.einsum('...jk,...ijk->...i', ginv, Chris)
    
    def local_diffusion(self, 
                        t:Array,
                        z:Array, 
                        )->Array:
        
        return self.M.Ginv(z)
    
    def TM_drift(self, 
                 t:Array,
                 z:Array, 
                 )->Array:
        
        return jnp.zeros_like(z)
    
    def TM_diffusion(self, 
                     t:Array,
                     z:Array, 
                     )->Array:
        
        return jnp.eye(self.dim)
    
    def local_diffusion(self, 
                        t:Array,
                        z:Array, 
                        )->Array:
        
        return self.M.Ginv(z)