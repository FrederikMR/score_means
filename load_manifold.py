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

from score_means.manifolds import nSphere, nEllipsoid, SPDN, HyperbolicSpace, FisherRaoGeometry, \
    HyperbolicParaboloid, T2, H2, nEuclidean, Paraboloid, Landmarks

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
        layers_s1 = [512, 512, 512]
        layers_s2 = [512, 512, 512]
        #layers_s1 = [512, 512, 512, 512, 512]
        #layers_s2 = [512, 512, 512, 512, 512]
        acts_s1 = [tanh]*len(layers_s1)
        acts_s2 = [tanh]*len(layers_s2)
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
    elif manifold == "SPDN":
        M = SPDN(N=dim)
        x0 = jnp.eye(dim).reshape(-1)
        layers = get_layers(manifold, dim)
    elif manifold == "HyperbolicSpace":
        M = HyperbolicSpace(dim=dim)
        val = jnp.ones(dim)
        x0 = jnp.zeros(dim+1)
        x0 = x0.at[:-1].set(val)
        x0 = x0.at[-1].set(jnp.sqrt(jnp.sum(x0)+1))
        layers = get_layers(manifold, dim)
    elif manifold == "Gaussian":
        M = FisherRaoGeometry(distribution="Gaussian")
        x0 = jnp.array([0.0, 0.0])
        layers = get_layers(manifold, dim)
    elif manifold == "HyperbolicParaboloid":
        M = HyperbolicParaboloid()
        x0 = jnp.zeros(2, dtype=jnp.float32)
        layers = get_layers(manifold, 2)
    elif manifold == "T2":
        M = T2(R=3.0, r=1.0)
        x0 = jnp.zeros(2, dtype=jnp.float32)
        layers = get_layers(manifold, 2)
    elif manifold == "Cylinder":
        M = T2(r=1.0)
        x0 = jnp.zeros(2, dtype=jnp.float32)
        layers = get_layers(manifold, 2)
    elif manifold == "H2":
        M = H2()
        x0 = jnp.zeros(2, dtype=jnp.float32)
        layers = get_layers(manifold, 2)
    elif manifold == "nEuclidean":
        M = nEuclidean(dim=dim)
        x0 = jnp.zeros(dim, dtype=jnp.float32)
        layers = get_layers(manifold, dim)
    elif manifold == "Paraboloid":
        M = Paraboloid()
        x0 = jnp.zeros(2, dtype=jnp.float32)
        layers = get_layers(manifold, 2)
    elif manifold == "Landmarks":
        M = Landmarks(N=dim, m=2)
        x0 = jnp.vstack((jnp.linspace(-5.0,5.0,M.N),jnp.linspace(0.0,0.0,M.N))).reshape(-1)
        layers = get_layers(manifold, dim*2)
    else:
        raise ValueError("The manifold is not implemented")

    return M, x0, layers[0], layers[1]