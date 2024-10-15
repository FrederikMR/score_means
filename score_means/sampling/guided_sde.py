#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 08:42:51 2024

@author: fmry
"""

#%% Modules

from .setup import *

from score_means.manifolds import RiemannianManifold

#%% Guided SDE

class GuidedSDE(ABC):
    def __init__(self,
                 SDE_fun:Callable,
                 phi_fun:Callable,
                 A:Callable = None,
                 )->None:
        
        self.SDE_fun = SDE_fun
        self.phi_fun = phi_fun
        
        # set default A if not specified
        self.A = A if A != None else \
             lambda z,v,w,*args: jnp.dot(v,jnp.linalg.solve(jnp.tensordot(X,X,(1,1)),w))
        self.A = A
        
        return
    
    def __str__(self)->str:
        
        return "Guided SDE Process"
    
    def sde_guided(self,
                   carry:Tuple[Array,Array,Array,Array],
                   step:Tuple[Array,Array,Array],
                   )->Array:
        z,log_likelihood,log_varphi,v,*cy = carry
        dt, t, dW = step
        
        (det,sto,X,*dcy) = self.SDE_fun((t,z,*cy),step)
        
        h = jax.lax.cond(t<self.T-dt/2,
                         lambda *_: self.phi(z,v,*cy)/(self.T-t),
                         lambda *_: jnp.zeros_like(self.phi((x,chart),v,*cy)),
                         )
        
        sto = jax.lax.cond(t < self.T-0.5*3*dt, # for Ito as well?
                           lambda *_: sto,
                           lambda *_: jnp.zeros_like(sto),
                           )

        ### likelihood
        dW_guided = (1.-.5*dt/(1-t))*dW+dt*h  # for Ito as well?
        sqrtCovx = sqrtCov(z,*cy) if sqrtCov is not None else X
        Cov = dt*jnp.tensordot(sqrtCovx,sqrtCovx,(1,1))
        Pres = jnp.linalg.inv(Cov)
        residual = jnp.tensordot(dW_guided,jnp.linalg.solve(Cov,dW_guided),(0,0))
        #residual = jnp.tensordot(dW_guided,jnp.tensordot(Pres,dW_guided,(1,0)),(0,0))
        log_likelihood = .5*(-dW.shape[0]*jnp.log(2*jnp.pi)-jnp.linalg.slogdet(Cov)[1]-residual)
        #log_likelihood = .5*(-dW.shape[0]*jnp.log(2*jnp.pi)+jnp.linalg.slogdet(Pres)[1]-residual)

        ## correction factor
        ytilde = jnp.tensordot(X,h*(self.T-t),1)
        tp1 = t+dt
        if integration == 'ito':
            xtp1 = x+dt*(det+jnp.tensordot(X,h,1))+sto
        elif integration == 'stratonovich':
            tx = x+sto
            xtp1 = x+dt*det+.5*(sto+sde((tp1,tx,chart,*cy),y)[1])
        xtp1chart = (xtp1,chart)
        Xtp1 = sde((tp1,xtp1,chart,*cy),y)[2]
        ytildetp1 = jax.lax.stop_gradient(jnp.tensordot(Xtp1,phi(xtp1chart,v,*cy),1)) # to avoid NaNs in gradient
        
        # set default A if not specified
        Af = A if A != None else \
             lambda x,v,w,*args: jnp.dot(v,jnp.linalg.solve(jnp.tensordot(X,X,(1,1)),w))

        #     add t1 term for general phi
        #     dxbdxt = theano.gradient.Rop((Gx-x[0]).flatten(),x[0],dx[0]) # use this for general phi
        t2 = jax.lax.cond(t<T-3*dt/2,
                          lambda *_: -Af(xchart,ytilde,det*dt,*cy)/(T-t),
                          # check det term for Stratonovich (correction likely missing)
                          lambda *_: 0.,
                          )
        t34 = jax.lax.cond(tp1<T-3*dt/2,
                           lambda *_: -(Af(xtp1chart,ytildetp1,ytildetp1,*cy)-Af(xchart,ytildetp1,ytildetp1,*cy)) / (
                           (T-tp1)),
                           lambda *_: 0.,
                           )
        log_varphi = t2 + t34

        return (det+jnp.dot(X,h),sto,X,log_likelihood,log_varphi,jnp.zeros_like(T),jnp.zeros_like(v),*dcy)
    
        guided = jit(lambda x,v,dts,dWs,*ys: integrate_sde(sde_guided,integrator_ito if integration == 'ito' else integrator_stratonovich,chart_update_guided,x[0],x[1],dts,dWs,0.,0.,jnp.sum(dts),M.update_coords(v,x[1])[0] if chart_update else v,*ys)[0:5])
        
    def _log_p_T(guided,A,phi,x,v,dW,dts,*ys):
        """ Monte Carlo approximation of log transition density from guided process """
        T = jnp.sum(dts)
        
        Cxv = jnp.sum(phi(x,M.update_coords(v,x[1])[0],*ys)**2)
        
        # sample
        log_varphis = jax.vmap(lambda dW: guided(x,v,dts,dW,*ys)[4][-1],1)(dW)
        
        log_varphi = jnp.log(jnp.mean(jnp.exp(log_varphis)))
        #(_,_,X,*_) = sde((T,v,chart,*cy),y)
        _logdetA = logdetA(x,*ys) if logdetA is not None else -2*jnp.linalg.slogdet(X)[1]
        log_p_T = .5*_logdetA-.5*x[0].shape[0]*jnp.log(2.*jnp.pi*T)-Cxv/(2.*T)+log_varphi
        return log_p_T
    
    log_p_T = partial(_log_p_T,guided,A,phi)

    neg_log_p_Ts = lambda *args: -jnp.mean(jax.vmap(lambda x,chart,w,dW,dts,*ys: log_p_T((x,chart),w,dW,dts,*ys),(None,None,0,0,None,*((None,)*(len(args)-5))))(*args))
