#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:11:07 2023

@author: fmry
"""

#%% Sources

#%% Modules

#jax
import jax.numpy as jnp

import numpy as np

import os

#argparse
import argparse

from load_manifold import load_manifold

from score_means.sampling import BrownianMotion

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="nEllipsoid",
                        type=str)
    parser.add_argument('--dim', default=2,
                        type=int)
    parser.add_argument('--N_sim', default=100,
                        type=int)
    parser.add_argument('--save_path', default='data/',
                        type=str)
    parser.add_argument('--T', default=0.5,
                        type=float)
    parser.add_argument('--dt_steps', default=1000,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% train for (x,y,t)

def generate_data()->None:
    
    args = parse_args()
    
    save_path = f"{args.save_path}{args.manifold}{args.dim}/"

    M, x0, *_ = load_manifold(args.manifold, args.dim)
    
    sampler = BrownianMotion(M, dt_steps=args.dt_steps, seed=args.seed)
    
    x0 = jnp.tile(x0, (args.N_sim, 1))
    xt = sampler(x0,T=0.5)[-1][-1]
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    np.savetxt(''.join((save_path, 'xs.csv')), xt, delimiter=",")
    
    return

#%% Main

if __name__ == '__main__':
        
    generate_data()