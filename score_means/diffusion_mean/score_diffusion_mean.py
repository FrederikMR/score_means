#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:29:52 2024

@author: fmry
"""

#%% Modules

from .setup import *

from score_means.manifolds import RiemannianManifold

#%% Score Gradients

class ScoreDiffusionMean(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 grady_fun:Tuple[Array, Array, Array],
                 ggrady_fun:Tuple[Array, Array, Array]=None,
                 gradt_fun:Tuple[Array, Array, Array]=None,
                 lr_rate:float=0.01,
                 beta1:float=0.9,
                 beta2:float=0.999,
                 eps:float=1e-08,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 method:str="Gradient"
                 )->None:
        
        self.M = M
        self.grady_fun = grady_fun
        if ggrady_fun is None:
            self.ggrady_fun = jacfwd(self.grady_fun, argnums=1)
        else:
            self.ggrady_fun = ggrady_fun
        if gradt_fun is not None:
            self.gradt_fun = gradt_fun
        else:
            self.gradt_fun = None
            
        self.lr_rate = lr_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol
        
        self.method = method
        
        return
            
    def __str__(self)->str:
        
        return "Score-based Diffusion t-mean object"
            
    def TM_hessy(self,
                x:Array,
                v:Array,
                h:Array,
                )->Array:
    
        val1 = self.M.TM_proj(x, h)
        val2 = v-self.M.TM_proj(x, v)
        val3 = jacfwd(self.M.TM_proj, argnums=0)(x,val2)
        
        return val1+val3
    
    def TM_grady(self,
                 x:Array,
                 v:Array,
                 )->Array:
    
        return self.M.TM_proj(x, v)
    
    def local_grady(self, 
                    x:Array, 
                    v:Array,
                    )->Array:
        
        Jf = self.M.Jf(x)

        return jnp.einsum('ij,i->j', Jf, v)
    
    def local_hessy(self, 
                    x:Array, 
                    v:Array, 
                    h:Array,
                    )->Array:
        
        val1 = self.M.Jf(x)
        val2 = jacfwd(lambda x1: self.M.Jf(x1))(x)
        
        term1 = jnp.einsum('jl,li,jk->ik', h, val1, val1)
        term2 = jnp.einsum('j,jik->ik', v, val2)
        
        return term1+term2
        
    def grady(self, 
              x:Array, 
              y:Array, 
              t:Array,
              )->Array:
        
        if self.M.intrinsic:
            return self.grady_fun(x,y,t)
        else:
            return self.M.TM_proj(y, self.grady_fun(x,y,t))
    
    def gradt(self, 
              x:Array, 
              y:Array, 
              t:Array,
              )->Array:
        
        s1 = self.grady_fun(x,y,t)
        s2 = self.ggrady_fun(x,y,t)
        
        if self.M.intrinsic:
            norm_s1 = jnp.dot(s1,s1)
            laplace_beltrami = jnp.trace(s2)+.5*jnp.dot(s1,jacfwd(lambda z: jnp.linalg.slogdet(self.M.G(z))[1])(y).squeeze())
        else:
            s2 = self.TM_hessy(y, s1, s2)
            norm_s1 = jnp.dot(s1,s1)
            laplace_beltrami = jnp.trace(self.TM_hessy(y, s1, s2))
    
        return 0.5*(laplace_beltrami+norm_s1)
    
    def grad(self,
             z:Array,
             q:Array,
             )->Array:
        
        t = sigmoid(q)
        gradz = jnp.mean(vmap(lambda x: self.grady(x,z,t))(self.X_obs), axis=0)
        gradq = jnp.mean(vmap(lambda x: self.gradt(x,z,t))(self.X_obs), axis=0)*grad(sigmoid)(q)
        
        return -jnp.hstack((gradz, gradq))
    
    def local_grad_step(self,
                   carry:Tuple[Array,Array,Array,int], 
                   )->Array:
        
        q, z, grad, idx = carry
        
        gradz = grad[:-1]
        gradq = grad[-1]
        
        q -= self.lr_rate*gradq
        z -= self.lr_rate*gradz
        
        grad = self.grad(z,q)
        
        return (q,z,grad,idx+1)
    
    def TM_grad_step(self,
                carry:Tuple[Array,Array,Array,int], 
                )->Array:
        
        q, z, grad, idx = carry
        
        gradz = grad[:-1]
        gradq = grad[-1]
        
        q -= self.lr_rate*gradq
        z = self.M.Exp(z, -self.lr_rate*gradz)
        
        grad = self.grad(z,q)
        
        return (q,z,grad,idx+1)
    
    def adam_update(self,
                    mt:Array,
                    vt:Array,
                    gt:Array,
                    t:int,
                    ):
        
        mt = self.beta1*mt+(1.-self.beta1)*gt
        vt = self.beta2*vt+(1-self.beta2)*(gt**2)
        
        mt_hat = mt/(1.-(self.beta1)**(t))
        vt_hat = vt/(1.-(self.beta2)**(t))
        
        step = self.lr_rate*mt_hat/(jnp.sqrt(vt_hat)+self.eps)
        
        return mt, vt, step
    
    def local_adam_step(self,
                   carry:Tuple[Array,Array,Array,Array,Array,int],
                   )->Array:
        
        q, z, gt, idx, mt, vt = carry
        
        mt, vt, step = self.adam_update(mt, vt, gt, idx)
        
        gradz = step[:-1]
        gradq = step[-1]
        
        q -= gradq
        z -= gradz

        gt = self.grad(z, q)
        
        return (q, z, gt, idx+1, mt, vt)
    
    def TM_adam_step(self,
                carry:Tuple[Array,Array,Array,Array,Array,int],
                )->Array:
        
        q, z, gt, idx, mt, vt = carry
        
        mt, vt, step = self.adam_update(mt, vt, gt, idx)
        
        gradz = step[:-1]
        gradq = step[-1]
        
        q -= gradq
        z = self.M.Exp(z, -gradz)
        
        gt = self.grad(z, q)
        
        return (q, z, gt, idx+1, mt, vt)
    
    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        q, z, grad, idx, *_ = carry
        
        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
        
    def __call__(self, 
                 x0:Array,
                 t0:Array,
                 X_obs:Array
                 )->Tuple[Array, Array]:
        
        q0 = jnp.log(t0/(1.-t0))
        self.X_obs = X_obs
        
        grad = self.grad(x0, q0)
        if self.method == "adam":
            if self.M.intrinsic:
                q, z, gt, mt, vt, idx = lax.while_loop(self.cond_fun, 
                                                       self.local_adam_step,
                                                       init_val=(q0, x0, gt, mt, vt, 1),
                                                       )
            else:
                q, z, gt, mt, vt, idx = lax.while_loop(self.cond_fun, 
                                                       self.TM_adam_step,
                                                       init_val=(q0, x0, gt, mt, vt, 1),
                                                       )
        elif self.method == "gradient":
            if self.M.intrinsic:
                q, z, grad, idx = lax.while_loop(self.cond_fun,
                                                 self.local_grad_step,
                                                 init_val=(q0,x0,grad,0),
                                                 )
            else:
                q, z, grad, idx = lax.while_loop(self.cond_fun,
                                                 self.TM_grad_step,
                                                 init_val=(q0,x0,grad,0),
                                                 )
        else:
            raise ValueError(f"Invalid method: {self.method}! Only gradient and adam supported.")

        t = sigmoid(q)
            
        return t, z, grad, idx