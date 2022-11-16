# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:14:29 2022

@author: mjaenada
"""
import numpy as np

def cef(y,ind,ranks,wl,ll):
    cey=win(y[ind,],wl,ll)
    return cey[ranks]


def win(y,wl,ll):

    r = np.convolve(y,np.ones(shape=(2*wl+1,)))
    r = r[wl:ll+wl]/(2*wl+1)
    r[:wl] = r[wl]
    r[ll-wl:ll] = r[ll-wl]
    return r 


def ace(x):
    
    # Alternating Conditional Expectation algorithm (ACE) to calculate optimal transformations
    # by fast boxcar averaging of rank-ordered data.
    # Program by: Henning U. Voss in MATLAB
    
    wl = 5
    oi = 100
    ii = 10
    ocrit = 2.2e-15
    icrit=1e-4
    
    dim, ll = x[1:,:].shape
    
    ind = np.zeros(shape=(dim+1,ll), dtype=int)
    ranks = np.zeros(shape=(dim+1,ll), dtype=int)
    
    for d in range(dim+1):
        ind[d,:] = np.argsort(x[d,:])
        ranks[d, ind[d,:]] = np.arange(ll)
    
    phi=(ranks-(ll-1)/2+1)/ np.sqrt(ll*(ll-1)/12)
    
    ieps=1
    oeps=1
    oi1=1
    ocrit1=1
    
    while oi1<=oi and ocrit1>ocrit :
        ii1=1
        icrit1=1
        while ii1<=ii and icrit1>icrit:
            for d in range(1,dim+1):
                sum0=0
                for dd in range(1,dim):
                    if dd !=d:
                        sum0 = sum0+phi[dd,:]
                phi[d,:] = cef(phi[0,:]-sum0,list(ind[d,:]),list(ranks[d,:]),wl,ll)
                
            icrit1=ieps
            if dim==1:
                sum0=phi[1,:]
            else:
                sum0=np.sum(phi[1:dim+1,:])
                
            ieps = np.sum((sum0-phi[0,:])**2)/ll
            icrit1 = np.abs(icrit1-ieps)
            ii1=ii1+1
        
        phi[0,:]=cef(sum0,ind[0,:],ranks[0,:],wl,ll)
        phi[0,:]=(phi[0,:]-np.mean(phi[0,:]))/np.std(phi[0,:])  
        ocrit1=oeps
        oeps=np.sum((sum0-phi[0,:])**2)/ll
        ocrit1=np.abs(ocrit1-oeps)
        
        oi1=oi1+1
        
    psi=np.corrcoef(phi[0,:],sum0)
    psi=psi[0,1]
    return psi
            
