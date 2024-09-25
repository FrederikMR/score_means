#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:38:54 2024

@author: fmry
"""

#%% Modules

from .setup import *

from .manifold import RiemannianManifold

#%% Code

class SPDN(RiemannianManifold):
    def __init__(self,
                 N:int=2,
                 eps:float=1e-10,
                 seed:int=2712,
                 )->None:

        self.N = N
        self.dim = N*(N-1)//2
        self.emb_dim = N**2
        self.eps = eps
        
        self.seed = seed
        self.key = jrandom.key(seed)
        
        super().__init__(f=self.f_stereographic, 
                         invf = self.invf_stereographic,
                         intrinsic=False)
        
        return
    
    def __str__(self)->str:
        
        return f"Symmetric Positive Definite Matrices of dimension {self.dim} embedded in R^{self.emb_dim}"
    
    def f_stereographic(self,
                        z:Array,
                        )->Array:
        
        s2 = jnp.sum(z**2)
        
        return self.params*jnp.hstack(((1-s2), 2*z))/(1+s2)

    def invf_stereographic(self, 
                           x:Array,
                           )->Array:
        
        x /= self.params
        
        x0 = x[0]
        return x[1:]/(1+x0)
    
    def TM_proj(self,
               x:Array,
               v:Array,
               )->Array:
        
        x = x.reshape(self.N, self.N)
        v = v.reshape(self.N, self.N)
        
        return (0.5*(v+v.T)).reshape(-1)
    
    def sample(self,
               N:int,
               x0:Array,
               sigma:float=1.0,
               )->Array:
    
        x0 = x0.reshape(self.N, self.N)-jnp.eye(self.N)
        
        key, subkey = jrandom.split(self.key)
        self.key = key
        z1 = jrandom.normal(subkey, shape=(N, self.N))
        
        key, subkey = jrandom.split(self.key)
        self.key = key
        s = jrandom.normal(subkey, shape=(N, self.N, self.N))
        Q,_ = jnp.linalg.qr(s)

        D = jnp.eye(self.N)+vmap(jnp.diag)(z1)
        
        return (x0+jnp.einsum('...ij,...jk,...lk->...il', Q, D, Q)).reshape(-1, 2*self.N)
    
    def dist(self,
             x:Array,
             y:Array
             )->Array:
        
        x = x.reshape(self.N, self.N)
        y = y.reshape(self.N, self.N)
        
        xinv = jnp.linalg.inv(x)
        xinv_half = self.sqrtm(xinv)
        
        inner = jnp.einsum('ij,jk,kl->il', xinv_half, y, xinv_half)
        log_mat = self.logm(inner)
        
        return jnp.linalg.norm(log_mat, ord='fro')
    
    def expm(self,
             x:Array,
             )->Array:
        
        x = x.reshape(self.N, self.N)
        
        U,S,V = jnp.linalg.svd(x)
        exp_val = jnp.einsum('ij,jk,kl->il', U, jnp.diag(jnp.exp(S)), V)
        
        return exp_val
    
    def logm(self,
             x:Array,
             )->Array:
        
        x = x.reshape(self.N, self.N)
        
        U,S,V = jnp.linalg.svd(x)
        log_val = jnp.einsum('ij,jk,kl->il', U, jnp.diag(jnp.log(S)), V)
        
        return log_val
    
    def sqrtm(self,
              x:Array,
              )->Array:
        
        x = x.reshape(self.N, self.N)
        
        U,S,V = jnp.linalg.svd(x)
        sqrt_val = jnp.einsum('ij,jk,kl->il', U, jnp.diag(jnp.sqrt(S)), V)
        
        return sqrt_val
    
    def Exp(self,
            x:Array,
            v:Array,
            T:float=1.0,
            )->Array:
        
        x = x.reshape(self.N, self.N)
        v = T*v.reshape(self.N, self.N)
        
        xinv = jnp.linalg.inv(x)
        x_half = self.sqrtm(x)
        xinv_half = self.sqrtm(xinv)
        #x_half = jnp.real(jscipy.linalg.sqrtm(x))
        #xinv_half = jnp.real(jscipy.linalg.sqrtm(xinv))
        
        inner = jnp.einsum('ij,jk,kl->il', xinv_half, v, xinv_half)
        res = jnp.einsum('ij,jk,kl->il', x_half, self.expm(inner), x_half)
        #res = jnp.einsum('ij,jk,kl->il', x_half, jscipy.linalg.expm(inner), x_half)
        
        return res.reshape(-1)
    
    def Log(self,
            x:Array,
            y:Array
            )->Array:
        
        x = x.reshape(self.N, self.N)
        y = y.reshape(self.N, self.N)
        
        xinv = jnp.linalg.inv(x)
        x_half = self.sqrtm(x)
        xinv_half = self.sqrtm(xinv)
        
        #x_half = jnp.real(jscipy.linalg.sqrtm(x))
        #xinv = jnp.linalg.inv(x)
        #xinv_half = jnp.real(jscipy.linalg.sqrtm(xinv))
        inner = jnp.einsum('ij,jk,kl->il', xinv_half, y, xinv_half)
        log_mat = self.logm(inner)
        
        res = jnp.einsum('ij,jk,kl->il', x_half, log_mat, x_half)
        
        return res.reshape(-1)
    
    def dot(self,
            x:Array,
            v:Array,
            w:Array
            )->Array:
        
        x = x.reshape(self.N, self.N)
        v = v.reshape(self.N, self.N)
        w = w.reshape(self.N, self.N)
        
        x_inv = jnp.linalg.inv(x)
        
        res = jnp.einsum('ij,jk,kl,lu->iu', x_inv, v, x_inv, w)
        
        return jnp.trace(res)
    
    def Geodesic(self,
                 x:Array,
                 y:Array,
                 t_grid:Array=None,
                 )->Array:
        
        if t_grid is None:
            t_grid = jnp.linspace(0.,1.,99, endpoint=False)[1:]
        
        x_s = x/self.params
        
        v = self.Log(x,y)
        
        gamma = self.params*vmap(lambda t: self.Exp(x_s, v,t))(t_grid)
        
        return jnp.vstack((x,gamma,y))