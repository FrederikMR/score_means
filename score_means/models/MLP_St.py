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

class MLP_St(hk.Module):
    def __init__(self,
                 layers:List=[500,500,500,500,500],
                 acts:List=[tanh,tanh,tanh,tanh,tanh],
                 init:hk.initializers = None,
                 )->None:
        super().__init__()

        self.layers = layers
        self.acts = acts
        self.init = init
    
    def model(self)->object:
        
        model = []
        for layer,act in zip(self.layers, self.acts):
            model.append(hk.Linear(layer, w_init=self.init, b_init=self.init))
            model.append(act)
            
        model.append(hk.Linear(1))

        return hk.Sequential(model)

    def __call__(self, 
                 x:Array,
                 )->Array:
        
        return self.model()(x).squeeze()