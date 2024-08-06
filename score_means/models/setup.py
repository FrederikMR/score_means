#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:51:28 2024

@author: fmry
"""

#%% Modules

from jax import Array
from jax import vmap
from jax.nn import tanh

import haiku as hk

from typing import List