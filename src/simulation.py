import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import statsmodels.api as sm
import random
#import math
#import itertools

#from scipy import stats, optimize
from scipy.linalg import sqrtm
#from scipy.optimize import NonlinearConstraint
#from sklearn.cross_decomposition import CCA
#from sympy.utilities.iterables import multiset_permutations

import canonical_estimation as cc

from get_metrics_functions import get_metrics, compare_metrics_cont
from generate_matrices_functions import *

"""
Archivo con funciones para generar matrices de ejemplos y simular.
En terminos de eficiencia, este metodo (sin transformar) parece funcionar mejor que step_estimation.
"""

###########################################################################
############################# SIMULATION ##################################
###########################################################################



########################### Simulate simple ###############################
###########################################################################
    
def loopfun(n, tau, numsim, generate_matrices_function, divergence, contamination, max_correlations, permutation_test = False):
    
    """
    Function that computes one complete iteration: Generates matrices, compute CCA for all values of tau
    and get metrics. Return an array with 5 measures for all taus.
    Permutation_test indicates if the incorrelation test is performance for dimensionality reduction
    
    """
    real = generate_matrices_function(n, numsim)

    #Variance-Covariance matrices
    real["SigmaX"] = np.cov(real["X"], rowvar=False, ddof=0)
    real["SigmaY"] = np.cov(real["Y"], rowvar=False, ddof=0)
    
    metrics_mat = np.zeros(shape=(6,max_correlations)) #All values of the array are zeros
    previous_correlations = dict()
    
    accessloop = True
    while (len(previous_correlations)<max_correlations):
        
        if permutation_test:
            accessloop = cc.contraste(real, tau, divergence, previous_correlations, nper=30)
            
        if accessloop:
            metrics_mat[0,len(previous_correlations)] = 1 #count the number of samples that accessloopf (whten permutation test is not rejected)
            #Maximize (= minimize -dist)
            result = cc.opt(real["X"],real["Y"],tau, real["SigmaX"], real["SigmaY"], divergence, previous_correlations)
            fitted_params = result.x
            metrics_mat[1:,len(previous_correlations)] = get_metrics(real, fitted_params)
            previous_correlations[str(len(previous_correlations))] = fitted_params
        else:
            break

    return metrics_mat

####################### Plot simple simulation ############################
###########################################################################
    
def plotcancorr(n, tau_list, numsim, generate_matrices_function, divergence, contamination, 
                max_correlations, permutation_test = False):
    
    """
    Function that computes one complete iteration: Generates matrices, compute CCA for all values of tau
    and get metrics. Return an array with 5 measures for all taus.
    Permutation_test indicates if the incorrelation test is performance for dimensionality reduction
    
    """
    real = generate_matrices_function(n, numsim,contamination)

    #Variance-Covariance matrices
    real["SigmaX"] = np.cov(real["X"], rowvar=False, ddof=0)
    real["SigmaY"] = np.cov(real["Y"], rowvar=False, ddof=0)
    
    metrics_mat = np.zeros(shape=(6,max_correlations)) #All values of the array are zeros
    previous_correlations = dict()
    
    accessloop = True
    for tau in tau_list:
        previous_correlations = dict()
        #while (len(previous_correlations)<max_correlations):
        for canonicalcorrelation in range(max_correlations):    
            
            if permutation_test:
                accessloop = cc.contraste(real, tau, divergence, previous_correlations, nper=30)
                
            if accessloop:
                metrics_mat[0,len(previous_correlations)] = 1 #count the number of samples that accessloopf (whten permutation test is not rejected)
                #Maximize (= minimize -dist)
                result = cc.opt(real["X"],real["Y"],tau, real["SigmaX"], real["SigmaY"], divergence, previous_correlations)
                fitted_params = result.x
                print(f'la correlacion con {tau} es {get_metrics(real, fitted_params)}')
                #metrics_mat[1:,len(previous_correlations)] = get_metrics(real, fitted_params)
                previous_correlations[str(len(previous_correlations))] = fitted_params
                
                # Plot comparaison of the original and estimates components
                Xlen = real["X"].shape[1]
                plt.subplot(121) #divide subplot in 1 row, 2 columns, start by 1
                plt.plot(np.dot(real["X"],fitted_params[:Xlen]), np.dot(real["Y"], fitted_params[Xlen:]), 'ro')
                plt.xlabel('x_canonical_estimate')
                plt.ylabel('y_canonical_estimate')
                plt.title(f'Scatterplot real y estimado para la componente {len(previous_correlations)} y tau = {tau}')
                
                # And now add something in the second part
                plt.subplot(122)
                plt.plot(np.dot(real["X"],real[f"a{canonicalcorrelation+1}"]), np.dot(real["Y"],real[f"b{canonicalcorrelation+1}"]), 'ro')
                plt.xlabel('x_canonical_real')
                plt.ylabel('y_canonical_real')
                # Show the graph
                plt.show()
                
                #plot scatterplot de las variables canonicas y sus estimadas
                plt.subplot(121) #divide subplot in 1 row, 2 columns, start by 1
                plt.plot(np.dot(real["X"],fitted_params[:Xlen]), np.dot(real["X"],real[f"a{canonicalcorrelation+1}"]), 'go')
                    
                # And now add something in the second part
                plt.subplot(122)
                plt.plot(np.dot(real["Y"],fitted_params[Xlen:]), np.dot(real["Y"],real[f"b{canonicalcorrelation+1}"]), 'go')
                plt.show()
                
                if contamination:
                    result = cc.opt(real["Xcont"],real["Ycont"],tau, 
                                    np.cov(real["SigmaX"], rowvar=False, ddof=0), np.cov(real["SigmaY"], rowvar=False, ddof=0),
                                    divergence,
                                    dict())
                    fitted_params = result.x
                    print(f'la correlacion con {tau} con contaminacion es {get_metrics(real, fitted_params)}')
                    # Plot comparaison of the original and estimates components
                    Xlen = real["X"].shape[1]
                    plt.subplot(121) #divide subplot in 1 row, 2 columns, start by 1
                    plt.plot(np.dot(real["Xcont"],fitted_params[:Xlen]), np.dot(real["Ycont"],fitted_params[Xlen:]), 'bo')
                    plt.xlabel('x_canonical_estimate_cont')
                    plt.ylabel('y_canonical_estimate_cont')
                    plt.title(f'Scatterplot real y estimado para la componente {len(previous_correlations)} y tau = {tau}')
                    
                    # And now add something in the second part
                    plt.subplot(122)
                    plt.plot(np.dot(real["Xcont"],real[f"a{canonicalcorrelation+1}"]), np.dot(real["Ycont"],real[f"b{canonicalcorrelation+1}"]), 'bo')
                    plt.xlabel('x_canonical_real_cont')
                    plt.ylabel('y_canonical_real_cont')
                    # Show the graph
                    plt.show() 
                    
                    #plot scatterplot de las variables canonicas y sus estimadas
                    plt.subplot(121) #divide subplot in 1 row, 2 columns, start by 1
                    plt.plot(np.dot(real["X"],fitted_params[:Xlen]), np.dot(real["X"],real[f"a{canonicalcorrelation+1}"]), 'go')
                    
                    # And now add something in the second part
                    plt.subplot(122)
                    plt.plot(np.dot(real["Y"],fitted_params[Xlen:]), np.dot(real["Y"],real[f"b{canonicalcorrelation+1}"]), 'go')
                    plt.show()
                    # Show the graph
                
                    # Show the graph
    
    
            else:
                break

    return get_metrics(real, fitted_params)


########################### Simulate robust ###############################
###########################################################################
    
def loopfun_robust(n, tau_list, numsim, generate_matrices_function, divergence, contamination, max_correlations, permutation_test = False):
    
    """
    Function that computes one complete iteration: Generates matrices, compute CCA for all values of tau
    and get metrics. Fitted with contaminated and uncontaminated data.
    Permutation_test indicates if the incorrelation test is performance for dimensionality reduction
    
    """
    real = generate_matrices_function(n, numsim, contamination)
    
    #Variance-Covariance matrices
    real["SigmaX"] = np.cov(real["X"], rowvar=False, ddof=0)
    real["SigmaY"] = np.cov(real["Y"], rowvar=False, ddof=0)
    
    cont = dict()
    cont["SigmaX"] = np.cov(real["Xcont"], rowvar=False, ddof=0)
    cont["SigmaY"] = np.cov(real["Ycont"], rowvar=False, ddof=0)
    
    ntau = len(tau_list)
    
    metrics_mat = np.zeros(shape=(6*ntau,max_correlations)) #All values of the array are zeros
    metrics_mat_cont = np.zeros(shape=(6*ntau,max_correlations)) #All values of the array are zeros
    
    comparaison_metrics = np.zeros(shape=(7*ntau,max_correlations)) #All values of the array are zeros
    accessloop = accessloop_cont = True
    permutation_test_cont = permutation_test
    
    init = False
    init_cont = False
    for t in range(ntau):
        
        tau = tau_list[t]
        previous_correlations = dict()
        previous_correlations_cont = dict()
        
        for canonicalcorrelation in range(max_correlations):
            
            if permutation_test:
                accessloop = cc.contraste(real, tau, divergence, previous_correlations, nper=30)
                permutation_test = accessloop
            
            # if accessloop:
                # metrics_mat[0+6*t,len(previous_correlations)] = 1 #count the number of samples that accessloopf (whten permutation test is not rejected)
                # #Maximize (= minimize -dist)
                # result = cc.opt(real["X"], real["Y"], tau, real["SigmaX"], real["SigmaY"],divergence, previous_correlations, init = init)
                # #resultados en escala transformada
                # fitted_params = result.x
                
                # metrics_mat[(1+6*t):6+6*t,len(previous_correlations)] = get_metrics(real, fitted_params, max_correlations)
                # previous_correlations[str(len(previous_correlations))] = fitted_params
                
                # if tau == 0:
                #     init = fitted_params
            #same for contaminated data
            if permutation_test_cont:
                accessloop_cont = cc.contraste(cont, tau, divergence, previous_correlations_cont, nper=30)
                permutation_test_cont = accessloop_cont
                
            if accessloop_cont:
                metrics_mat_cont[0+6*t,len(previous_correlations_cont)] = 1 #count the number of samples that accessloopf (whten permutation test is not rejected)
                #Maximize (= minimize -dist)
                result_cont = cc.opt2(real["Xcont"],real["Ycont"], tau, cont["SigmaX"], cont["SigmaY"], divergence, previous_correlations_cont, init = init_cont)
                fitted_params_cont = result_cont
            
                metrics_mat_cont[1+6*t:6+6*t,len(previous_correlations_cont)] = get_metrics(real, fitted_params_cont, max_correlations)
                previous_correlations_cont[str(len(previous_correlations_cont))] = fitted_params_cont
                
                if tau == 0:
                    init_cont = fitted_params_cont
                
            # if (accessloop and accessloop_cont):
    
                # #metrics comparaison
                # comparaison_metrics[0+7*t,len(previous_correlations)-1] = 1
                # comparaison_metrics[1+7*t:7+7*t,len(previous_correlations)-1] = compare_metrics_cont(real, cont, fitted_params, fitted_params_cont, max_correlations)
            
            if not (accessloop or accessloop_cont):
                break
            
            

    return np.concatenate((metrics_mat, metrics_mat_cont, comparaison_metrics), axis=0)
"""  
def simulate(n, tau_list, generate_matrices_function, contamination = 0, loopfun = loopfun,
             max_correlations=1, permutation_test=False, R=10):
    
    metrics_taulist = dict()
    for tau in tau_list:
        applyfun = np.vectorize(lambda x: loopfun(n,tau,x, generate_matrices_function, contamination, max_correlations, permutation_test),otypes=[np.ndarray])
        metrics = applyfun(np.arange(R))
        metrics = np.stack(metrics, 0).astype(np.float64)
        metrics = np.apply_along_axis(np.mean, 0, metrics)
        metrics_taulist[str(tau)] = pd.DataFrame(metrics)
    print(metrics_taulist)
    
    #return dataframe with results
    return pd.concat(metrics_taulist, axis=1)
"""
# def simulate(n, tau_list, generate_matrices_function, contamination = 0, loopfun = loopfun, max_correlations=1, permutation_test=False, R=10):
    
#     metrics_taulist = dict()
#     for tau in tau_list:
#         applyfun = np.vectorize(lambda x: loopfun(n,tau,x, generate_matrices_function,  contamination, max_correlations, permutation_test),otypes=[np.ndarray])
#         metrics = applyfun(np.arange(R))
#         metrics = np.stack(metrics, 0).astype(np.float64)
#         metrics = np.apply_along_axis(np.mean, 0, metrics)
#         metrics_taulist[str(tau)] = pd.DataFrame(metrics)
#     print(metrics_taulist)
#     return pd.concat(metrics_taulist, axis=1)

def simulate(n, tau_list, generate_matrices_function, divergence = "RP", contamination = 0, loopfun = loopfun, max_correlations=1, permutation_test=False, R=10):
    
    #metrics_taulist = dict()
    
    applyfun = np.vectorize(lambda x: loopfun(n,tau_list,x, generate_matrices_function,  divergence, contamination, max_correlations, permutation_test),otypes=[np.ndarray])
    m = applyfun(np.arange(R))
    metrics = np.apply_along_axis(np.mean, 0, m)
    
    #Now I form the DataFrame dividing in taus
    subbloques = [6,6,7]
    bloques = [0,6,12,19]
    results_all = []
    
    for canvar in range(max_correlations):
        
        results = np.zeros(shape=(19,len(tau_list)))
        
        for t in range(len(tau_list)):
            for lb in range(1,len(bloques)):
                start = len(tau_list)*bloques[lb-1] + subbloques[lb-1]*t
                end =  start + bloques[lb] -bloques[lb-1] #bloques[lb] + subbloques[lb-1]*t
                results[bloques[lb-1]:bloques[lb],t] = metrics[start:end,canvar].reshape((end-start,))
                
        results = pd.DataFrame(data=results)
        results.index = [ 'converge1', 'corr_between', 'corr_real_a','corr_real_b', 'L2a', 'L2b',
                      'converge2', 'corrcont_between', 'corrcont_real_a', 
                      'corrcont_real_b',  'L2acont', 'L2bcont', 'converge3',
                      'L2a', 'L2b', 'N2a', 'N2b', 'N2ar', 'N2br']
        results_all.append(results)
        
    return m,results_all



n=100
tau_list=[0,0.2,0.4,0.6]
generate_matrices_function = generate_matrices5
divergence = "DPD"
contamination = 0.2
loopfun= loopfun_robust
max_correlations=2
R=10
permutation_test = False
         
# it,res= simulate(n=n,tau_list=tau_list, generate_matrices_function = generate_matrices_function, 
#                     divergence = "DPD", contamination = 0.1, loopfun= loopfun,
#                     max_correlations=max_correlations, R=R, permutation_test = False)
# res
# pd.concat(res).to_csv("DPD01.csv")


# res1= simulate(n=n,tau_list=tau_list, 
#                     generate_matrices_function = generate_matrices_function, 
#                     divergence = "RP", contamination = 0.1,
#                     loopfun= loopfun,  max_correlations=max_correlations, R=R, permutation_test = False)
# res1
# pd.concat(res1).to_csv("RP01")

# res2= simulate(n=n,tau_list=tau_list, generate_matrices_function = generate_matrices_function, 
#                     divergence = "DPD", contamination = 0.15, loopfun= loopfun,
#                     max_correlations=max_correlations, R=R, permutation_test = False)
# res2
# pd.concat(res2).to_csv("DPD015")


# res3= simulate(n=n,tau_list=tau_list, 
#                     generate_matrices_function = generate_matrices_function, 
#                     divergence = "RP", contamination = 0.15,
#                     loopfun= loopfun,  max_correlations=max_correlations, R=R, permutation_test = False)
# res3
# pd.concat(res3).to_csv("RP015")



# res4= simulate(n=n,tau_list=tau_list, generate_matrices_function = generate_matrices4, 
#                     divergence = "DPD", contamination = 0.1, loopfun= loopfun,
#                     max_correlations=2, R=R, permutation_test = False)
# res4
# pd.concat(res4).to_csv("DPD01_4")


# res5= simulate(n=n,tau_list=tau_list, 
#                     generate_matrices_function = generate_matrices4, 
#                     divergence = "RP", contamination = 0.1,
#                     loopfun= loopfun,  max_correlations=2, R=R, permutation_test = False)
# res5
# pd.concat(res5).to_csv("RP01_4")

# res6= simulate(n=n,tau_list=tau_list, generate_matrices_function = generate_matrices4, 
#                     divergence = "DPD", contamination = 0.15, loopfun= loopfun,
#                     max_correlations=2, R=R, permutation_test = False)
# res6
# pd.concat(res6).to_csv("DPD015_4")


# res7= simulate(n=n,tau_list=tau_list, 
#                     generate_matrices_function = generate_matrices4, 
#                     divergence = "RP", contamination = 0.15,
#                     loopfun= loopfun,  max_correlations=2, R=R, permutation_test = False)
# res7
# pd.concat(res7).to_csv("RP015_4")



# # for numsim in [11]:
# #     #tau=0
# #     print(f'para {tau_list} el error es {plotcancorr(n, tau_list, numsim, generate_matrices_function, contamination, max_correlations, permutation_test = False)}')
# #     #tau=0.5
# #     #print(f'para {tau} el error es {plotcancorr(n, tau, numsim, generate_matrices_function, contamination, max_correlations, permutation_test = False)}')
    