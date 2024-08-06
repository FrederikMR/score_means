#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:32:35 2024

@author: fmry
"""

#%% Modules

from .setup import *

from score_means.manifolds import RiemannianManifold


#%% First Order Scores

class S1Loss(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 method="ism",
                 samples:int=1,
                 seed:int=2712,
                 )->None:
        
        if not (method in ['ism', 'dsm', 'dsmvr', 'ssm', 'ssmvr']):
            raise ValueError("Invalid Loss Fun")
        
        self.M = M
        self.method = method
        self.samples = samples
        self.seed = seed
        
        if self.method == "ism":
            self.loss = self.ism
        elif self.method == "dsm":
            self.loss = self.dsm
        elif self.method == "dsmvr":
            self.loss = self.dsmvr
        elif self.method == "ssm":
            self.loss = self.ssm
        elif self.method == "ssmvr":
            self.loss = self.ssmvr
        
        self.key = jrandom.key(seed)
        
        return
    
    def __str__(self)->str:
        
        return f"Score Matching Loss Fun with loss function {self.method}"
    
    def StdNormal(self, 
                  d:int, 
                  num:int
                  )->Array:
        
        key, subkey = jrandom.split(self.key)
        self.key = key

        return jrandom.normal(subkey, shape=(num, d)).squeeze()
    
    def ism(self,
            s1_model:Callable[[Array, Array, Array], Array],
            x0:Array,
            xt:Array,
            t:Array,
            dW:Array,
            dt:Array,
            )->Array:
        
        loss_s1 = s1_model(x0,xt,t.reshape(-1,1))
        norm2s = jnp.sum(loss_s1*loss_s1, axis=1)
        
        divs = vmap(lambda x,y,t: self.M.div(y, lambda y1: s1_model(x,y1,t)))(x0,xt,t)
        
        return jnp.mean(norm2s+2.0*divs)
    
    def dsm(self,
            s1_model:Callable[[Array, Array, Array], Array],
            x0:Array,
            xt:Array,
            t:Array,
            dW:Array,
            dt:Array,
            )->Array:
        
        s1 = s1_model(x0,xt,t.reshape(-1,1))
        
        loss = dW/dt.reshape(-1,1)+s1
        
        return jnp.mean(jnp.sum(loss, axis=-1))
    
    def dsmvr(self,
              s1_model:Callable[[Array, Array, Array], Array],
              x0:Array,
              xt:Array,
              t:Array,
              dW:Array,
              dt:Array,
              )->Array:
        
        s1 = s1_model(x0,xt,t.reshape(-1,1))
        s1p = s1_model(x0,x0,t.reshape(-1,1))
        
        l1_loss = dW/dt.reshape(-1,1)+s1
        l1_loss = 0.5*jnp.einsum('...i,...i->...', l1_loss,l1_loss)
        var_loss = jnp.einsum('...i,...i->...', s1p,dW)/dt+jnp.einsum('...i,...i->...', dW,dW)/(dt**2)
        
        return jnp.mean(l1_loss-var_loss)
    
    def ssm(self,
            s1_model:Callable[[Array, Array, Array], Array],
            x0:Array,
            xt:Array,
            t:Array,
            dW:Array,
            dt:Array,
            )->Array:
        
        s1 = s1_model(x0,xt,t.reshape(-1,1))
        v = self.StdNormal(d=x0.shape[-1],num=self.M*x0.shape[0]).reshape(-1,x0.shape[-1])
        
        val = lambda x,y,t,v: grad(lambda y0: jnp.dot(v,s1_model(x,y0,t)))(y)
        
        vs1 = vmap(val)(x0,xt,t,v)
        J = 0.5*jnp.einsum('...j,...j->...',v,s1)**2
        
        return jnp.mean(J+jnp.einsum('...i,...i->...', vs1, v))
    
    def ssmvr(self,
              s1_model:Callable[[Array, Array, Array], Array],
              x0:Array,
              xt:Array,
              t:Array,
              dW:Array,
              dt:Array,
              )->Array:
        
        s1 = s1_model(x0,xt,t.reshape(-1,1))
        v = self.StdNormal(d=x0.shape[-1],num=self.M*x0.shape[0]).reshape(-1,x0.shape[-1])
        
        val = lambda x,y,t,v: grad(lambda y0: jnp.dot(v,s1_model(x,y0,t)))(y)
        
        vs1 = vmap(val)(x0,xt,t,v)
        J = 0.5*jnp.einsum('...j,...j->...',s1,s1)
        
        return jnp.mean(J+jnp.einsum('...i,...i->...', vs1, v))
    
    def __call__(self,
                 s1_model:Callable[[Array, Array, Array], Array],
                 x0:Array,
                 xt:Array,
                 t:Array,
                 dW:Array,
                 dt:Array,
                 )->Array:
        
        return self.loss(s1_model, x0, xt, t, dW, dt)
    
#%% Second Order Scores

class S2Loss(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 method="dsm",
                 )->None:
        
        if not (method in ['dsm', 'dsmvr']):
            raise ValueError("Invalid Loss Fun")
        
        self.M = M
        self.method = method
        
        if self.method == "dsm":
            self.loss = self.dsm
        elif self.method == "dsmvr":
            self.loss = self.dsmvr
        elif self.method == "dsmdiag":
            self.loss = self.dsmdiag
        elif self.method == "dsmddiagvr":
            self.loss = self.dsmdiagvr
        
        return
    
    def __str__(self)->str:
        
        return f"Score Matching Loss Fun with loss function {self.method}"
    
    def dsm(self,
            s1_model:Callable[[Array, Array, Array], Array],
            s2_model:Callable[[Array, Array, Array], Array],
            x0:Array,
            xt:Array,
            t:Array,
            dW:Array,
            dt:Array,
            )->Array:
        
        eye = jnp.eye(dW.shape[-1])
        s1 = lax.stop_gradient(s1_model(x0,xt,t.reshape(-1,1)))
        s2 = s2_model(x0,xt,t.reshape(-1,1))
        
        loss_s2 = s2+jnp.einsum('...i,...j->...ij', s1, s1)+(eye-jnp.einsum('...i,...j->ij', dW, dW)/dt.reshape(-1,1))/dt.reshape(-1,1)
        
        return jnp.mean(jnp.sum(jnp.square(loss_s2), axis=-1))
    
    def dsmdiag(self,
                s1_model:Callable[[Array, Array, Array], Array],
                s2_model:Callable[[Array, Array, Array], Array],
                x0:Array,
                xt:Array,
                t:Array,
                dW:Array,
                dt:Array,
                )->Array:
        
        s1 = lax.stop_gradient(s1_model(x0,xt,t.reshape(-1,1)))
        s2 = s2_model(x0,xt,t.reshape(-1,1))

        s1 = lax.stop_gradient(s1_model(x0,xt,t.reshape(-1,1)))
        s2 = s2_model(x0,xt,t.reshape(-1,1))
        
        loss_s2 = vmap(jnp.diag)(s2)+s1*s1+(1.0-dW*dW/dt.reshape(-1,1))/dt.reshape(-1,1)
        
        return jnp.mean(jnp.sum(jnp.square(loss_s2), axis=-1))
    
    def dsmvr(self,
              s1_model:Callable[[Array, Array, Array], Array],
              s2_model:Callable[[Array, Array, Array], Array],
              x0:Array,
              xt:Array,
              t:Array,
              dW:Array,
              dt:Array,
              )->Array:
        
        eye = jnp.eye(dW.shape[-1])
        s1 = lax.stop_gradient(s1_model(x0,x0,t.reshape(-1,1)))
        s1p = lax.stop_gradient(s1_model(x0,xt,t.reshape(-1,1)))
        s2 = s2_model(x0,x0,t.reshape(-1,1))
        s2p = s2_model(x0,xt,t.reshape(-1,1))
        
        psi = s2+jnp.einsum('i,j->ij', s1, s1)
        psip = s2p+jnp.einsum('i,j->ij', s1p, s1p)
        diff = (eye-jnp.einsum('i,j->ij', dW, dW)/dt)/dt
        
        loss1 = psip**2
        loss2 = 2.*diff*(psip-psi)
        
        loss_s2 = loss1+loss2

        return 0.5*jnp.mean(jnp.sum(loss_s2, axis=-1))
    
    def dsmdiagvr(self,
                  s1_model:Callable[[Array, Array, Array], Array],
                  s2_model:Callable[[Array, Array, Array], Array],
                  x0:Array,
                  xt:Array,
                  t:Array,
                  dW:Array,
                  dt:Array,
                  )->Array:

        s1 = lax.stop_gradient(s1_model(x0,x0,t.reshape(-1,1)))
        s1p = lax.stop_gradient(s1_model(x0,xt,t.reshape(-1,1)))
        s2 = s2_model(x0,x0,t.reshape(-1,1))
        s2p = s2_model(x0,xt,t.reshape(-1,1))
        
        psi = vmap(jnp.diag)(s2)+s1*s1
        psip = vmap(jnp.diag)(s2p)+s1p*s1p
        diff = (1.0-dW*dW/dt)/dt
        
        loss1 = psip**2
        loss2 = 2.*diff*(psip-psi)
        
        loss_s2 = loss1+loss2

        return 0.5*jnp.mean(jnp.sum(loss_s2, axis=-1))

    def __call__(self,
                 s1_model:Callable[[Array, Array, Array], Array],
                 s2_model:Callable[[Array, Array, Array], Array],
                 x0:Array,
                 xt:Array,
                 t:Array,
                 dW:Array,
                 dt:Array,
                 )->Array:
        
        return self.loss(s1_model, s2_model, x0, xt, t, dW, dt)