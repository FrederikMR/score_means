#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:27:51 2024

@author: fmry
"""

#%% Modules

from jax import Array
from jax import vmap, grad, lax, jacfwd

from jax.nn import sigmoid

import jax.numpy as jnp

from typing import Callable, Tuple

from abc import ABC