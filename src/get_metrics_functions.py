# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:14:33 2022

@author: mjaenada
"""
import numpy as np
import pandas as pd


def normalize(v):
    """
    computes normalized version of a vector
    """
    v = np.array(v)
    v = v/np.sqrt(np.sum(v**2))
    return v

def get_metrics(real, estimated, max_correlations = 1):
    
    """
    Get metrics: correlation between the estimated canonical vectors, real and estimated components.
    """
    Xlen = real["X"].shape[1]
    #Ylen = real["Y"].shape[1]
    
    estimated_a = normalize(estimated[:Xlen])
    estimated_b = normalize(estimated[Xlen:])
    
    corres = []
    corres.append(abs(np.corrcoef(np.dot(real["X"], estimated_a ), np.dot(real["Y"],estimated_b) )[0,1]))
    
    #como no se que variable canonica coje, me quedo con la mayor
    cc1 = []
    #L2ar= []
    cc2 = []
    #L2br = []
    
    for c in range(max_correlations):
        cc1.append(abs(np.corrcoef(np.dot(real["X"], estimated_a ), np.dot(real["X"], normalize(real[f'a{c+1}'])) )[0,1] ))
        cc2.append(abs(np.corrcoef(np.dot(real["Y"], estimated_b ), np.dot(real["Y"], normalize(real[f'b{c+1}'])) )[0,1] ))
        
        #L2ar.append( np.sqrt(np.sum( np.dot(np.identity(estimated_a.shape[0]) - normalize(real[f'a{c+1}']) @ normalize(real[f'a{c+1}']).T, estimated_a)**2 ) )  )
        #L2br.append( np.sqrt(np.sum( np.dot(np.identity(estimated_b.shape[0]) - normalize(real[f'b{c+1}']) @ normalize(real[f'b{c+1}']).T, estimated_b)**2 ) )  )
    #true component
    c1 = np.argmax(cc1)
    c2 = np.argmax(cc2)
    
    corres.append(np.max(cc1))
    corres.append(np.max(cc2))
    
    corres.append( np.sqrt(np.sum( np.dot(np.identity(estimated_a.shape[0]) - normalize(real[f'a{c1+1}']) @ normalize(real[f'a{c1+1}']).T, estimated_a)**2 ) )  )
    corres.append( np.sqrt(np.sum( np.dot(np.identity(estimated_b.shape[0]) - normalize(real[f'b{c2+1}']) @ normalize(real[f'b{c2+1}']).T, estimated_b)**2 ) )  )
    
    return corres

def get_metrics_transformed(real, X, Y, sqrt_SigmaX, sqrt_SigmaY, estimated, max_correlations = 1):
    
    """
    Get metrics: correlation between the estimated canonical vectors, real and estimated components.
    """
    Xlen = real["X"].shape[1]
    #Ylen = real["Y"].shape[1]
    
    estimated_a = normalize(estimated[:Xlen])
    estimated_b = normalize(estimated[Xlen:])
    
    corres = []
    corres.append(abs(np.corrcoef(np.dot(X, estimated_a ), np.dot(Y,estimated_b) )[0,1]))
    
    #como no se que variable canonica coje, me quedo con la mayor
    cc1 = []
    #L2ar= []
    cc2 = []
    #L2br = []
    
    
    
    #for c in range(max_correlations):
    c=0
    while f'a{c+1}' in real.keys():
        
        cc1.append(abs(np.corrcoef(np.dot(X, estimated_a ), np.dot(X, normalize(np.dot(sqrt_SigmaX,real[f'a{c+1}']))) )[0,1] ))
        cc2.append(abs(np.corrcoef(np.dot(Y, estimated_b ), np.dot(Y, normalize(np.dot(sqrt_SigmaY,real[f'b{c+1}']))) )[0,1] ))
        
        #L2ar.append( np.sqrt(np.sum( np.dot(np.identity(estimated_a.shape[0]) - normalize(real[f'a{c+1}']) @ normalize(real[f'a{c+1}']).T, estimated_a)**2 ) )  )
        #L2br.append( np.sqrt(np.sum( np.dot(np.identity(estimated_b.shape[0]) - normalize(real[f'b{c+1}']) @ normalize(real[f'b{c+1}']).T, estimated_b)**2 ) )  )
        
        c=c+1
   
    #true component
    #c1 = np.argmax(cc1)
    c2 = np.argmax(cc2)
    c1 = c2
    
    corres.append(np.max(cc1))
    corres.append(np.max(cc2))
    
    corres.append( np.sqrt(np.sum( np.dot(np.identity(estimated_a.shape[0]) - normalize(np.dot(sqrt_SigmaX,real[f'a{c1+1}'])) @ normalize(np.dot(sqrt_SigmaX,real[f'a{c1+1}'])).T, estimated_a)**2 ) )  )
    corres.append( np.sqrt(np.sum( np.dot(np.identity(estimated_b.shape[0]) - normalize(np.dot(sqrt_SigmaY,real[f'b{c2+1}'])) @ normalize(np.dot(sqrt_SigmaY,real[f'b{c2+1}'])).T, estimated_b)**2 ) )  )
    
    return corres


def compare_metrics_cont(real, real_cont, estimated, estimated_cont, max_correlations=1):
    
    """
    Get metrics about contamination influence on the canconical vectors.
    """
    Xlen = real["X"].shape[1]
    Ylen = real["Y"].shape[1]
    
    estimated_a = normalize(estimated[:Xlen]).reshape((Xlen,1))
    estimated_b = normalize(estimated[Xlen:]).reshape((Ylen,1))
    
    estimated_cont_a = normalize(estimated_cont[:Xlen]).reshape((Xlen,1))
    estimated_cont_b = normalize(estimated_cont[Xlen:]).reshape((Ylen,1))
    
    
    L2a = np.sqrt(np.sum( np.dot(np.identity(estimated_a.shape[0]) - estimated_a @ estimated_a.T, estimated_cont_a)**2 ) ) 
    L2b = np.sqrt(np.sum( np.dot(np.identity(estimated_b.shape[0]) - estimated_b @ estimated_b.T, estimated_cont_b)**2 ) )
    
    
    
    #otra forma de medir distancia
    #CALCULARLO PERO SOBRE LAS VARIABLES CANANONICAS NO LOS COEFICIENTES
    
    #distancia en norma
    N2a = np.sqrt(np.sum((estimated_a-estimated_cont_a)**2))
    N2b = np.sqrt(np.sum((estimated_b-estimated_cont_b)**2))
    #distancia con los reales
    N2ar = []
    N2br = []
    for c in range(max_correlations):
        N2ar.append(np.sqrt(np.sum((normalize(real[f'a{c+1}'])-estimated_cont_a)**2)))
        N2br.append(np.sqrt(np.sum((normalize(real[f'b{c+1}'])-estimated_cont_b)**2)))
    N2ar = np.min(N2ar)
    N2br = np.min(N2br)
    
    corres = [L2a, L2b, N2a, N2b, N2ar, N2br]
    
#    corres.append(abs(np.corrcoef(np.dot(real["X"], estimated_a ).T, np.dot(real["Y"],estimated_b).T )[0,1] ))
#    corres.append(abs(np.corrcoef(np.dot(real["X"], estimated_a ).T , np.dot(real["X"], real["a"]).T  )[0,1] ))
#    corres.append(abs(np.corrcoef( np.dot(real["Y"], estimated_b ).T , np.dot(real["Y"], real["b"]).T  )[0,1] ))
#    
#    corres.append(abs(np.corrcoef(np.dot(real["X"], estimated_cont_a ).T , np.dot(real["Y"],estimated_cont_b).T )[0,1] ))
#    corres.append(abs(np.corrcoef(np.dot(real["X"], estimated_cont_a ).T , np.dot(real["X"], real["a"]).T )[0,1] ))
#    corres.append(abs(np.corrcoef( np.dot(real["Y"], estimated_cont_b ).T , np.dot(real["Y"], real["b"]) )[0,1] ))

    return corres