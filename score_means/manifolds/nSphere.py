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