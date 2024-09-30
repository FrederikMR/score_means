#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:43:21 2024

@author: fmry
"""

#%% Modules

from .setup import *

from score_means.manifolds import RiemannianManifold

#%% MLP Score

class MLP_S1(hk.Module):
    def __init__(self,
                 M:RiemannianManifold,
                 layers:List=[500,500,500,500,500],
                 acts:List=[tanh,tanh,tanh,tanh,tanh],
                 init:hk.initializers = None,
                 )->None:
        super().__init__()
        
        self.M = M
        self.layers = layers
        self.acts = acts
        self.init = init
    
    def model(self)->object:
        
        model = []
        for layer,act in zip(self.layers, self.acts):
            model.append(hk.Linear(layer, w_init=self.init, b_init=self.init))
            model.append(act)

        return hk.Sequential(model)

    def __call__(self, 
                 x:Array,
                 )->Array:
        
        encoded = self.model()(x)
        if self.M.intrinsic:
            return hk.Linear(self.M.dim)(encoded)
        else:
            x_point = x.T[:self.M.emb_dim].T
            if x.ndim == 1:
                #return hk.Linear(self.M.emb_dim)(encoded)
                return self.M.TM_proj(x_point, hk.Linear(self.M.emb_dim)(encoded))
            else:
                #return hk.Linear(self.M.emb_dim)(encoded)
                return vmap(self.M.TM_proj)(x_point, hk.Linear(self.M.emb_dim)(encoded))