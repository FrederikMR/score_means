#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:52:32 2024

@author: fmry
"""

#%% Modules

import haiku as hk

import os

#argparse
import argparse

from load_manifold import load_manifold

from score_means.models import MLP_S1, MLP_St
from score_means.score_matching import BrownianSampler
from score_means.score_matching import train_s1, train_st

#%% Arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="nSphere",
                        type=str)
    parser.add_argument('--dim', default=2,
                        type=int)
    parser.add_argument('--loss_type', default="ism",
                        type=str)
    parser.add_argument('--load_model', default=0,
                        type=int)
    parser.add_argument('--T_sample', default=0,
                        type=int)
    parser.add_argument('--t0', default=0.1,
                        type=float)
    parser.add_argument('--gamma', default=1.0,
                        type=float)
    parser.add_argument('--sigma', default=100.0,
                        type=float)
    parser.add_argument('--train_net', default="s1",
                        type=str)
    parser.add_argument('--T', default=1.0,
                        type=float)
    parser.add_argument('--lr_rate', default=0.001,
                        type=float)
    parser.add_argument('--epochs', default=200000,
                        type=int)
    parser.add_argument('--warmup_epochs', default=1000,
                        type=int)
    parser.add_argument('--x0_samples', default=10, #32
                        type=int)
    parser.add_argument('--xt_samples', default=32, #32
                        type=int)
    parser.add_argument('--t_samples', default=100, #32
                        type=int)
    parser.add_argument('--dt_steps', default=100,
                        type=int)
    parser.add_argument('--save_step', default=10,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% Training

def train_score()->None:
    
    args = parse_args()

    T_sample_name = (args.T_sample == 1)*"T"
    st_path = f"scores/{args.manifold}{args.dim}/st/"
    s1_path = f"scores/{args.manifold}{args.dim}/s1{T_sample_name}_{args.loss_type}/"
    #s2_path = f"scores/{args.manifold}{args.dim}/s2{T_sample_name}_{args.s2_loss_type}/"
    #s1s2_path = f"scores/{args.manifold}{args.dim}/s1s2{T_sample_name}_{args.s2_loss_type}/"
    
    if not os.path.exists(s1_path):
        os.makedirs(s1_path)
    if not os.path.exists(st_path):
        os.makedirs(st_path)
    
    M, x0, layers, acts = load_manifold(args.manifold, args.dim)
    layers_s1, layers_s2 = layers
    acts_s1, acts_s2 = acts
    
    s1_model = hk.transform(lambda x: MLP_S1(M=M, 
                                             layers=layers_s1,
                                             acts = acts_s1,
                                             init = None,
                                             )(x))
    st_model = hk.transform(lambda x: MLP_St(layers=layers_s1,
                                             acts = acts_s1,
                                             init = None,
                                             )(x))
    
    data_generator = BrownianSampler(M, 
                                     x0,
                                     x0_samples=args.x0_samples,
                                     xt_samples=args.xt_samples,
                                     t_samples=args.t_samples,
                                     dt_steps=args.dt_steps,
                                     T=args.T,
                                     T_sample=args.T_sample,
                                     t0=args.t0,
                                     sigma=args.sigma,
                                     seed=args.seed,
                                     )
    
    if args.train_net == "s1":
        train_s1(M=M,
                 model=s1_model,
                 generator=data_generator,
                 state = None,
                 lr_rate = args.lr_rate,
                 epochs = args.epochs,
                 warmup_epochs = args.warmup_epochs,
                 save_step = args.save_step,
                 optimizer = None,
                 save_path = s1_path,
                 loss_type = args.loss_type,
                 seed = args.seed,
                 )
    elif args.train_net == "st":
        train_st(M=M,
                 model=st_model,
                 generator=data_generator,
                 state=None,
                 lr_rate=args.lr_rate,
                 epochs=args.epochs,
                 warmup_epochs=args.warmup_epochs,
                 save_step=args.save_step,
                 optimizer=None,
                 save_path=st_path,
                 seed=args.seed,
                 )
    
    return

#%% Main

if __name__ == '__main__':
        
    train_score()