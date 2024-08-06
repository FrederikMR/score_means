#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:37:41 2024

@author: fmry
"""

#%% Modules

from jax import Array, tree_leaves, tree_map, tree_flatten, tree_unflatten, device_get

from jax import jit, value_and_grad, vmap, grad
from jax import lax

import jax.numpy as jnp
import jax.random as jrandom

import numpy as np

import random

import os

import pickle

import tensorflow as tf
import tensorflow_datasets as tfds

import haiku as hk

import optax

from typing import NamedTuple, Dict, Callable, Tuple

from abc import ABC