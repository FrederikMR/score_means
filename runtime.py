#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:52:36 2024

@author: fmry
"""

#%% Sources

#https://jax.readthedocs.io/en/latest/faq.html

#%% Modules

from jax import Array

import jax.numpy as jnp
import jax.random as jrandom

import haiku as hk

import pandas as pd

import timeit

import argparse

from typing import List

from load_manifold import load_manifold

from score_means.score_matching import load_model
from score_means.diffusion_mean import ScoreDiffusionMean
from score_means.models import MLP_S1, MLP_St

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="nSphere",
                        type=str)
    parser.add_argument('--dim', default=2,
                        type=int)
    parser.add_argument('--s1_loss_type', default="ism",
                        type=str)
    parser.add_argument('--s2_loss_type', default="dsm",
                        type=str)
    parser.add_argument('--dt_approx', default="st",
                        type=str)
    parser.add_argument('--t0', default=0.01,
                        type=float)
    parser.add_argument('--lr_rate', default=0.01,
                        type=float)
    parser.add_argument('--score_iter', default=1000,
                        type=int)
    parser.add_argument('--bridge_iter', default=100,
                        type=int)
    parser.add_argument('--tol', default=1e-4,
                        type=float)
    parser.add_argument('--t_init', default=0.2,
                        type=float)
    parser.add_argument('--estimate', default="diffusion_mean",
                        type=str)
    parser.add_argument('--benchmark', default=0,
                        type=int)
    parser.add_argument('--data_path', default='data/',
                        type=str)
    parser.add_argument('--save_path', default='table/estimates/',
                        type=str)
    parser.add_argument('--score_path', default='scores/',
                        type=str)
    parser.add_argument('--timing_repeats', default=5,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% Estimate Diffusion Mean

def estimate_diffusion_mean(DiffusionMean:object, 
                            x0:Array,
                            t0:Array,
                            X_obs:Array,
                            x_true:Array=None,
                            t_true:Array=None,
                            ):
    
    args = parse_args()
    
    method = {} 
    t, x = DiffusionMean(x0, t0, X_obs)
    print("\t-Estimate Computed")
    timing = []
    timing = timeit.repeat(lambda: DiffusionMean(x0, t0, X_obs)[1].block_until_ready(), 
                           number=args.number_repeats, 
                           repeat=args.timing_repeats)
    print("\t-Timing Computed")
    timing = jnp.stack(timing)
    
    
    method['mu_time'] = jnp.mean(timing)
    method['std_time'] = jnp.std(timing)
    method['t'] = t
    method['x'] = x
    if x_true is not None:
        method['x_error'] = jnp.linalg.norm(x-x_true)
    else:
        method['x_error'] = None
    if t_true is not None:
        method['t_error'] = jnp.linalg.norm(t-t_true)
    else:
        method['t_error'] = None
    
    return method

#%% Estimate Frechet mean

def estimate_frechet_mean(FrechetMean:object, 
                          x0:Array,
                          X_obs:Array,
                          x_true:Array=None,
                          ):
    
    args = parse_args()
    
    method = {} 
    t, x = FrechetMean(x0, X_obs)
    print("\t-Estimate Computed")
    timing = []
    timing = timeit.repeat(lambda: FrechetMean(x0, X_obs)[1].block_until_ready(), 
                           number=args.number_repeats, 
                           repeat=args.timing_repeats)
    print("\t-Timing Computed")
    timing = jnp.stack(timing)
    
    
    method['mu_time'] = jnp.mean(timing)
    method['std_time'] = jnp.std(timing)
    method['t'] = t
    method['x'] = x
    if x_true is not None:
        method['x_error'] = jnp.linalg.norm(x-x_true)
    else:
        method['x_error'] = None
    
    return method

#%% Diffusion mean run-time

def runtime_diffusion_mean()->None:
    
    args = parse_args()
    rng_key = jrandom.PRNGKey(args.seed)
    
    data_path = f"{args.data_path}{args.manifold}{args.dim}/"
    st_path = f"scores/{args.manifold}{args.dim}/st/"
    s1_path = f"scores/{args.manifold}{args.dim}/s1_{args.s1_loss_type}/"
    
    M, x0, layers, acts = load_manifold(args.manifold, args.dim)
    layers_s1, layers_s2 = layers
    acts_s1, acts_s2 = acts
    X_obs = jnp.array(pd.read_csv(''.join((data_path, 'xs.csv')), header=None))
    
    s1_state = load_model(s1_path)
    
    s1_model = hk.transform(lambda x: MLP_S1(M=M, 
                                             layers=layers_s1,
                                             acts = acts_s1,
                                             init = None,
                                             )(x))
    if args.dt_approx == "s1":
        st_fun = None
    elif args.dt_approx == "st":
        st_state = load_model(st_path)
        st_model = hk.transform(lambda x: MLP_St(layers=layers_s1,
                                                 acts = acts_s1,
                                                 init = None,
                                                 )(x))
        st_fun = lambda x,y,t: st_model.apply(st_state.params, rng_key, jnp.hstack((x,y,t)))
    
    ScoreMean = ScoreDiffusionMean(M, 
                                   grady_fun = lambda x,y,t: s1_model.apply(s1_state.params, rng_key, jnp.hstack((x,y,t))),
                                   gradt_fun = st_fun,
                                   lr_rate = args.lr_rate,
                                   max_iter=args.score_iter,
                                   tol=args.tol,
                                   method="gradient",
                                   )
    
    t,x, grad_val, idx = ScoreMean(X_obs[0],
                                   args.t_init,
                                   X_obs)
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(10,10))
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_obs[:,0], X_obs[:,1], X_obs[:,2])
    ax.scatter(x[0],x[1],x[2], s=1000)
    
    from jax import vmap, grad
    from jax.nn import sigmoid
    q0 = jnp.log(t/(1.-t))
    print(jnp.mean(vmap(lambda y: ScoreMean.gradt(y,x,t))(ScoreMean.X_obs), axis=0))
    print(grad(sigmoid)(q0))
    print(jnp.mean(vmap(lambda y: ScoreMean.gradt(y,x,t))(ScoreMean.X_obs), axis=0)*grad(sigmoid)(q0))
    
    print(t)
    print(x)
    
    from jax import grad
    
    if M.intrinsic:
        grady_fun=lambda x,y,t: grad(lambda x1,y1,t1: jnp.log(M.heat_kernel(x1,y1,t1)), 
                                     argnums=1)(x,y,t)
    else:
        grady_fun=lambda x,y,t: M.TM_proj(y, 
                                          grad(lambda x1,y1,t1: jnp.log(M.heat_kernel(x1,y1,t1)), 
                                                                        argnums=1)(x,y,t))
    gradt_fun=lambda x,y,t: grad(lambda x1,y1,t1: jnp.log(M.heat_kernel(x1,y1,t1)), 
                                 argnums=2)(x,y,t)
    
    ScoreMean = ScoreDiffusionMean(M, 
                                   grady_fun = grady_fun,
                                   gradt_fun = gradt_fun,
                                   lr_rate = args.lr_rate,
                                   max_iter=args.score_iter,
                                   tol=args.tol,
                                   method="gradient",
                                   )
    
    t,x, grad_val, idx = ScoreMean(X_obs[0],
                                   args.t_init,
                                   X_obs)
    
    from jax import vmap
    print(jnp.mean(vmap(lambda y: ScoreMean.gradt(y,x,t))(ScoreMean.X_obs), axis=0))
    print(grad(sigmoid)(q0))
    print(jnp.mean(vmap(lambda y: ScoreMean.gradt(y,x,t))(ScoreMean.X_obs), axis=0)*grad(sigmoid)(q0))
    
    print(t)
    print(x)
    
    fig = plt.figure(figsize=(10,10))
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_obs[:,0], X_obs[:,1], X_obs[:,2])
    ax.scatter(x[0],x[1],x[2], s=1000)
    
    return
    
    
#%% Frechet mean run-time

def runtime_frechet_mean()->None:
    
    args = parse_args()
    
    
#%% Main

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.estimate == "diffusion_mean":
        runtime_diffusion_mean()
    else:
        runtime_frechet_mean()


