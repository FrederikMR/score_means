#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:06:20 2024

@author: fmry
"""

#%% Modules

from jax import Array
from jax import lax, vmap

import jax.numpy as jnp

import jax.random as jrandom

from typing import Tuple

from abc import ABC