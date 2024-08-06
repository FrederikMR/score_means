#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:34:51 2024

@author: fmry
"""

#%% Modules

from jax import Array
import jax.numpy as jnp
from jax.nn import tanh

from typing import List, Tuple

from score_means.manifolds import nSphere, nEllipsoid

#%% Layers

def get_layers(manifold:str, dim:int)->Tuple[List, List, List, List]:
    
    if manifold == "Euclidean":
        layers_s1 = [128, 128, 128]
        layers_s2 = [128,128,128]
        acts_s1 = [tanh, tanh, tanh]
        acts_s2 = [tanh, tanh, tanh]
        #if dim < 15:
        #    layers_s1 = [128, 128, 128]
        #    layers_s2 = [32,32,32]
        #else:
        #    layers_s1 = [512, 512, 512, 512, 512]
        #    layers_s2 = [128, 128, 128, 128, 128]
    elif manifold == "Sphere":
        layers_s1 = [512, 512, 512, 512, 512]
        layers_s2 = [512, 512, 512, 512, 512]
        acts_s1 = [tanh, tanh, tanh, tanh, tanh]
        acts_s2 = [tanh, tanh, tanh, tanh, tanh]
    else:
        layers_s1 = [512, 512, 512]
        layers_s2 = [512, 512, 512]
        acts_s1 = [tanh, tanh, tanh]
        acts_s2 = [tanh, tanh, tanh]
        
    return (layers_s1, layers_s2), (acts_s1, acts_s2)

#%% Load Manifold

def load_manifold(manifold:str, dim:int=None)->Tuple[object, Array, List]:

    if manifold == "nSphere":
        M = nSphere(dim=dim)
        x0 = jnp.zeros(M.emb_dim, dtype=jnp.float32)
        x0 = x0.at[-1].set(1.0)
        layers = get_layers(manifold, dim)
    elif manifold == "nEllipsoid":
        params = jnp.linspace(0.5,1.0,dim+1)
        M = nEllipsoid(dim=dim, params=params)
        x0 = jnp.zeros(M.emb_dim, dtype=jnp.float32)
        x0 = x0.at[-1].set(1.0)
        x0 *= params
        layers = get_layers(manifold, dim)
    else:
        raise ValueError("The manifold is not implemented")

    return M, x0, layers[0], layers[1]