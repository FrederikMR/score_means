#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:47:44 2024

@author: fmry
"""

#%% Modules

from jax import jacfwd, vmap, grad
from jax import lax
from jax import Array

import jax.numpy as jnp

from typing import Callable

from abc import ABC
