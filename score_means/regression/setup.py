#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:29:23 2024

@author: fmry
"""

#%% Modules

from jax import Array
from jax import vmap, grad
from jax import lax

import jax.numpy as jnp

#JAX Optimization
from jax.example_libraries import optimizers

from abc import ABC
from typing import Callable, Tuple 