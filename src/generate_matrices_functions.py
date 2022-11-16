# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:16:50 2022

@author: mjaenada
"""

import numpy as np
import random
from scipy.linalg import sqrtm

def generate_matrices(n,seed,contamination=0):
    
    Xlen, Ylen = 8,3
    
    np.random.seed(seed)
    X = np.random.normal(0, 1, size=(n, Xlen))
    
    Y =np.empty(shape=(n,Ylen))
    Y[:,0] = np.square(2*X[:,0]+ X[:,1]+ X[:,2]) + 0.5*np.random.normal(0, 1, size=(n, ))
    Y[:,1:Ylen] = np.random.normal(0, 1, size=(n,Ylen-1))
    
    a = [2,1,1] + [0 for i in range(Xlen-3)]
    b = [1] + [0 for i in range(Ylen-1)]
    
    return {"X": X,  "Y": Y,  "a1": a, "b1": b }


def generate_matrices2(n,seed,contamination=0):
    
    """
    Function to  generate matrices, parameter contamination indicates contamination level of Ycont
    Fixed seed for reproducibility.
    """
    
    Xlen, Ylen = 8,3
    
    np.random.seed(seed)
    X = np.empty(shape=(n,Xlen))
    X[:,:3] = np.random.normal(0, 1, size=(n, 3))
    X[:,3] = np.random.chisquare(df = 7, size=n)
    X[:,4] = np.random.standard_t(df = 5, size=n)
    X[:,5] = np.random.f(dfnum = 3, dfden=12, size=n)
    X[:,6:8] = np.random.normal(0, 1, size=(n, 2))
    
    Y = np.empty(shape=(n,Ylen))
    Y[:,0] = np.square(2*X[:,0]+ X[:,1]+ 2*X[:,2])
    Y[:,1] = np.random.normal(0, 1, size=n)
    Y[:,2] = np.random.standard_t(df = 9, size=n)
    
    ncont = int(np.floor(contamination*n))
    sample = random.sample(range(n),ncont)
    contpos = np.array([p in sample for p in range(n)])
    nclean = int(n-ncont)
    
    Y[~contpos,0]=   Y[~contpos,0]+ 0.05*np.random.normal(0, 1, size=(nclean, ))
    Ycont = np.copy(Y)
    Y[contpos,0]=   Y[contpos,0]+ 0.05*np.random.normal(0, 1, size=(ncont, ))
    Ycont[contpos,0]=   Ycont[contpos,0] + np.random.uniform(0, 50, size=(ncont, ))
  
    
#    ncont = int(np.floor(contamination*n))
#    nclean = int(n-ncont)
#    
#    Y[:nclean,0]=   Y[:nclean,0]+ np.sqrt(0.5)*np.random.normal(0, 1, size=(nclean, ))
#    Ycont = np.copy(Y)
#    Y[nclean:,0]=   Y[nclean:,0]+ np.sqrt(0.5)*np.random.normal(0, 1, size=(ncont, ))
#    Ycont[nclean:,0]=   Ycont[nclean:,0] + np.random.uniform(0, 50, size=(ncont, ))
#  
    a = [2,1,2] + [0 for i in range(Xlen-3)]
    b = [1]+ [0 for i in range(Ylen-1)]
    
    return {"X": X, "Xcont": X, "Y": Y, "Ycont":Ycont, "a1": a, "b1": b }


def generate_matrices4(n,seed=314,contamination=0):
    
    """
    Function to  generate matrices, parameter contamination indicates contamination level of Ycont
    Fixed seed for reproducibility.
    """
    
    Xlen, Ylen = 8,3
    
    np.random.seed(seed)
    X = np.empty(shape=(n,Xlen))
    X[:,:3] = np.random.normal(0, 1, size=(n, 3))
    X[:,3] = np.random.chisquare(df = 7, size=n)
    X[:,4] = np.random.standard_t(df = 5, size=n)
    X[:,5] = np.random.f(dfnum = 3, dfden=12, size=n)
    X[:,6:8] = np.random.normal(0, 1, size=(n, 2))
    
    Y = np.empty(shape=(n,Ylen))
    Y[:,0] = np.square(2*X[:,0]+ X[:,1]+ X[:,2])
    Y[:,1] = X[:,1]-X[:,2]#np.random.normal(0, 1, size=n)
    Y[:,2] = np.random.standard_t(df = 9, size=n) #np.random.normal(0, 1, size=n)
    
    #np.random.seed(seed)
    sample = np.random.binomial(1, contamination, size=n)
    sample = sample.astype(bool)

    Y[:,0]=   Y[:,0]+ 0.5*np.random.normal(0, 1, size=(n, ))
    Y[:,1]=   Y[:,1]+ 0.2*np.random.normal(0, 1, size=(n, ))
    
    Ycont = np.copy(Y)
    Xcont = np.copy(X)
    
    Ycont[sample,1] =   Y[sample,0] #+ np.random.uniform(0, 50, size=(ncont, ))
    Ycont[sample,0] =   Y[sample,1] #+ np.random.uniform(0, 80, size=(ncont, ))
   
  
    a1 = [0,1,-1,0,0] + [0 for i in range(Xlen-5)]
    b1 = [0,1]+ [0 for i in range(Ylen-2)]
    
    a2 = [2,1,1] + [0 for i in range(Xlen-3)]
    b2 = [1,0]+ [0 for i in range(Ylen-2)]
    
    return {"X": X, "Xcont": Xcont, "Y": Y, "Ycont":Ycont, "a1": a1, "b1": b1, "a2":a2, "b2":b2 }


def generate_matrices5(n,seed=314,contamination=0):
    
    """
    Function to  generate matrices, parameter contamination indicates contamination level of Ycont
    Fixed seed for reproducibility.
    """
    
    Xlen, Ylen = 4,3
    
    np.random.seed(seed)
    X = np.empty(shape=(n,Xlen))
    X[:,:3] = np.random.normal(0, 1, size=(n, 3))
    X[:,3] = np.random.uniform(0, np.pi,size=(n, ))
    #X[:,4] = np.random.standard_t(df = 5, size=n)
    #X[:,5] = np.random.f(dfnum = 3, dfden=12, size=n)
    #X[:,4:6] = np.random.normal(0, 1, size=(n, 2))
    
    Y = np.empty(shape=(n,Ylen))
    Y[:,0] = (X[:,0]- 2*X[:,1]+ X[:,2])**2
    Y[:,1] = np.cos(X[:,3])
    Y[:,2] = np.random.standard_t(df = 9, size=n) 
    #Y[:,3] = np.random.normal(0, 1, size=n)
    
    #np.random.seed(seed)
    sample = np.random.binomial(1, contamination, size=n)
    sample = sample.astype(bool)

    Y[:,0]=   Y[:,0]+ 0.1*np.random.normal(0, 1, size=(n, ))
    Y[:,1]=   Y[:,1]+ 0.1*np.random.uniform(-1,1, size=(n,))
    
    Ycont = np.copy(Y)
    Xcont = np.copy(X)
    
    Ycont[sample,1] =   Y[sample,0] #+ np.random.uniform(0, 50, size=(ncont, ))
    Ycont[sample,0] =   Y[sample,1] #+ np.random.uniform(0, 80, size=(ncont, ))
   
  
    a1 = [1,-2,1,0] 
    b1 = [1,0] + [0]
    
    a2 = [0,0,0,1] 
    b2 = [0,1] + [0]
    
    return {"X": X, "Xcont": Xcont, "Y": Y, "Ycont":Ycont, "a1": a1, "b1": b1, "a2":a2, "b2":b2 }


def generate_matrices3(n,seed=314,contamination=0):
    
    """
    Function to  generate matrices, parameter contamination indicates contamination level of Ycont
    Fixed seed for reproducibility.
    """
    
    Xlen, Ylen = 8,3
    seed = seed+314
    np.random.seed(seed)
    X = np.random.normal(0, 1, size=(n, 8))
    
    Y = np.empty(shape=(n,Ylen))
    Y[:,0] = 2*X[:,0]+ X[:,1]+ X[:,2]  #np.square(2*X[:,0]+ X[:,1]+ X[:,2])
    Y[:,1] =  np.random.normal(0, 1, size=n) #X[:,1]-X[:,3]#np.random.normal(0, 1, size=n)
    Y[:,2] =  np.random.normal(0, 1, size=n) #np.random.standard_t(df = 9, size=n)
    
    np.random.seed(seed+10)
    ncont = int(np.floor(contamination*n))
    sample = random.sample(range(n),ncont)
    contpos = np.array([p in sample for p in range(n)])
    
    np.random.seed(seed+10**2)
    Y[~contpos,0]=   Y[~contpos,0]+ 0.5*np.random.normal(0, 1, size=(int(n-ncont), ))
    
    Ycont = np.copy(Y)
    Xcont = np.copy(X)
    
    np.random.seed(seed+10**3)
    Y[contpos,0]=   Y[contpos,0]+ 0.5*np.random.normal(0, 1, size=(ncont, ))
    np.random.seed(seed+10**3)
    Ycont[contpos,0] =   Ycont[contpos,0] + np.random.uniform(0, 50, size=(ncont, ))

    a = [2,1,1,0,0] + [0 for i in range(Xlen-5)]
    b = [1,0]+ [0 for i in range(Ylen-2)]
    
    # # Whitened
    
    # # #Variance-Covariance matrices
    # sqrt_SigmaX = sqrtm(np.cov(X, rowvar=False, ddof=0))
    # sqrt_SigmaY = sqrtm(np.cov(Y, rowvar=False, ddof=0))
    
    # #whitened
    # X = (X - X.mean(axis=0)[np.newaxis,:]).dot(np.linalg.inv(sqrt_SigmaX))
    # Y = (Y - Y.mean(axis=0)[np.newaxis,:]).dot(np.linalg.inv(sqrt_SigmaY))
    

    
    # #Variance-Covariance matrices of contaminated data
    # sqrt_SigmaXcont = sqrtm(np.cov(Xcont, rowvar=False, ddof=0))
    # sqrt_SigmaYcont = sqrtm(np.cov(Ycont, rowvar=False, ddof=0))
    
    # #transformed vector
    # a = np.dot(sqrt_SigmaXcont, a)
    # b = np.dot(sqrt_SigmaYcont, b)
    
    # #whitened
    # Xcont = (Xcont - Xcont.mean(axis=0)[np.newaxis,:]).dot(np.linalg.inv(sqrt_SigmaXcont))
    # Ycont = (Ycont - Ycont.mean(axis=0)[np.newaxis,:]).dot(np.linalg.inv(sqrt_SigmaYcont))
    
    return {"X": X, "Xcont": Xcont, "Y": Y, "Ycont":Ycont, "a1": a, "b1": b }