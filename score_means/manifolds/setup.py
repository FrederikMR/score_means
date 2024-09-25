#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:47:44 2024

@author: fmry
"""

#%% Modules

from jax import jacfwd, vmap, grad, hessian
from jax import lax
from jax import Array

import jax.numpy as jnp

import jax.random as jrandom

import jax.scipy as jscipy

from typing import Callable, Dict, Tuple

from abc import ABC
