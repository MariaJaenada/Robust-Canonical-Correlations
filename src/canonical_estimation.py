# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:47:46 2022

@author: mjaenada
"""

import numpy as np
import statsmodels.api as sm

import random

from scipy import stats, optimize
from scipy.linalg import sqrtm
from scipy.optimize import NonlinearConstraint

from get_metrics_functions import *
from ace import ace

###########################################################################
######################### FITNESS FUNCTIONS ###############################
###########################################################################

def normalize(v):
    
    """
    Computes normalized version of a vector
    
    Parameters
    -----------
    v:  numpy array or list
        vector to be normalized
    
    Returns
    -----------
    v : numpy array 
        normalized vector
    """
    v = np.array(v)
    v = v/np.sqrt(np.sum(v**2))
    return v

def est_uni (w,n):
    
    """
    Unidimensional estimation of the density function.
    It uses implemented functions in sm package. 
    If the computation is hard it can be computed directly.
    
    Parameters
    -----------
    w: numpy array
        (n,1)-dimensional array of first canonical component
    n: number of sample size
    
    Returns
    -----------
    Estimated density at points w1,w2
    
    """
    #Estimated standard deviation
    s = stats.tstd(w)

    #Soft parameter proposed in Scott 
    h = (4/3)**(1/5)*s*n**(-1/5)

    #Model call and fitness
    #modelo_kde_uni = sm.nonparametric.KDEUnivariate(w)
    #modelo_kde_uni.fit(kernel="gau", bw=h, fft = True)
    
    est=[]
    for x in w:
        est.append(1/(n*h)*np.sum(1/np.sqrt(2*np.pi)*np.exp(-((x-w)/h)**2/2)))
    
    return est #modelo_kde_uni


def est_bid (w1, w2,n):
    
    """
    Bidimensional estimation of the density function.
    It uses implemented functions in sm package. 
    If the computation is hard it can be computed directly.
    To do: generalize to n-dimensional setting
    
    Parameters
    -----------
    w1: numpy array
        array of first canonical component in X: aX
    w2: numpy array
        array of first canonical component in Y: bY
    n: number of sample size
    
    Returns
    -----------
    Estimated density at points w1,w2
    
    """
    #Standard deviation from the first and second sample
    s1 = stats.tstd(w1)
    s2 = stats.tstd(w2)

    #Soft parameter proposed in Scott 
    h1 = s1*n**(-1/6)
    h2 = s2*n**(-1/6)
    
    w1 = w1.reshape((w1.shape[0],1))
    w2 = w2.reshape((w2.shape[0],1))
    
    #Model definition and data fitness
    #
    #modelo_kde_bid = sm.nonparametric.KDEMultivariate(data=[w1,w2], var_type='cc', bw=[h1,h2])
    est = []
    for i in range(n):
        est.append(1/(n*h1*h2)*np.sum( 1/(2*np.pi)*np.exp(-((w1[i]-w1)/h1)**2/2 - ((w2[i]-w2)/h2)**2/2) ) )
                   
    return est #modelo_kde_bid


def distance (c,X,Y, tau, divergence):
    
    """
    Compute RP/DPD  with tuning parameter tau between two canonical vectors aX, bY.
    
    Parameters
    ---------
    c: numpy array or list
        concatenated array with all canonical components,
        c=[a,b] for simplicity on the minimization process
    X,Y: numpy array 
        First data matrix
    tau: Float
        Tuning parameter. tau >0
    divergence: string
            "RP" or "DPD"  
    
    Returns
    ----------
    dist: float
        negative distance between estimated densities of aX and bY
    """
    
    n, Xlen = X.shape
    
    a = c[:Xlen]
    b = c[Xlen:]
    
    #Canonical correlations
    aX = np.dot(X,a).reshape(n)
    bY = np.dot(Y,b).reshape(n)

    #Estimated density functions and their evaluation at the sample
    est_uniu = est_uni(aX,n)
    est_univ = est_uni(bY,n)
    est_biduv = est_bid(aX, bY, n)
    
    du = np.array(est_uniu) #est_uniu.evaluate(aX)
    dv = np.array(est_univ) #est_univ.evaluate(bY)
    duv = np.array(est_biduv) #np.apply_along_axis(est_biduv.pdf, 0, np.stack((aX,bY)))
    
    #return the distance KL for tau=0, else pseudodistance Renyi
    
    
    if not tau:
        dist=n*np.mean(np.log(duv)- np.log(du*dv)) 
    else:
        if divergence == "DPD":
            dist = n*(np.mean(du**tau)*np.mean(dv**tau) - (1+1/tau)*np.mean(du**tau*dv**tau) + (1/tau)*np.mean(duv**tau)) #DPD
            #dist=dist/tau
        else:
            dist = n*((1/tau)*np.log(np.mean(duv**tau)) - ((tau+1)/tau)*np.log(np.mean(du**tau*dv**tau))  + np.log(np.mean(du**tau)*np.mean(dv**tau)))
            #dist=dist/tau
    
    return float(-dist)

#Some auxiliar function to compute cuadratic producs for variance and covariances
def cov_can_a (c, Xlen, cov, ck):
    
    """
    First group of components variance-covariances

    Parameters
    ----------
    c : numpy array or list
        contain elements of all canonical vectors
    Xlen : int
        number of variables in first group
    cov : numpy array
        Covariate matrix
    ck : numpy array or list
        contain elements of all canonical vectors

    Returns
    -------
    var_can_a : float
        variance or covariance between Xlen first components of c and ck.

    """
    a = c[:Xlen]
    ak = ck[:Xlen]
    var_can_a = (a.T).dot(cov).dot(ak)
    return var_can_a


def cov_can_b (c, Xlen, cov, ck):
    
    """
    Second group of components variance-covariances

    Parameters
    ----------
    c : numpy array or list
        contain elements of all canonical vectors
    Xlen : int
        number of variables in first group
    cov : numpy array
        Covariate matrix
    ck : numpy array or list
        contain elements of all canonical vectors

    Returns
    -------
    var_can_b : float
        variance or covariance between after Xlen  components of b and bk.

    """
    
    b = c[Xlen:]
    bk = ck[Xlen:]
    var_can_b = (b.T).dot(cov).dot(bk)
    return var_can_b

def initial_guess(SigmaX, SigmaY, previous_correlations=dict()):

    """
    Initial start point satisfying the quadratic constraints.
    For the incorrelation constraint, we project the vectors to orthogonal space. 
    
    Parameters
    -----------
    SigmaX, SigmaY: numpy array matrices
                    Covariance matrices
    previous_correlations: dict
                    previous canonical vectors. Default ditc()
    """
    
    Xlen = SigmaX.shape[1]
    Ylen = SigmaY.shape[1]
    
    if len(previous_correlations):
        
        sqrt_SigmaX = sqrtm(SigmaX)
        sqrt_SigmaY = sqrtm(SigmaY)
        A = np.empty(shape=(Xlen,len(previous_correlations)))
        B = np.empty(shape=(Ylen,len(previous_correlations)))
        
        for component in range(len(previous_correlations)):
            A[:,component] = np.dot(sqrt_SigmaX, previous_correlations[str(component)][:Xlen])
            B[:,component] = np.dot(sqrt_SigmaY, previous_correlations[str(component)][Xlen:])
        A = A @ A.T
        B = B @ B.T
        
        Au, _, _ = np.linalg.svd(A, full_matrices=True) #quiza As, Avh (otros resultados) sean necesarios en otro momento 
        Bu, _, _ = np.linalg.svd(B, full_matrices=True) #por ahora no los guardo
        
        inita = Au[:,len(previous_correlations)]
        initb = Bu[:,len(previous_correlations)]
        
        init = np.concatenate((inita,initb))
        
    else:
        inita = np.random.normal(0,0.5,Xlen)
        n_inita = np.sqrt(np.dot(SigmaX, inita).dot(inita))
        inita = inita/n_inita
       
        initb = np.random.normal(0,0.5, Ylen)
        n_initb = np.sqrt(np.dot(SigmaY, initb).dot(initb))
        initb = initb/n_initb
        
        init = np.concatenate((inita,initb)) 
        
    return init


def opt_canonical(X,Y, tau, SigmaX, SigmaY, divergence = "RP", previous_correlations = dict(), init=False, Repetitions = 10):
    
    """
    Optimization function. Return the optimized function.

    Parameters
    ----------
    X : numpy array
        Data group 1
    Y : numpy array
        Data group 2
    tau : float
        divergence tuning parameter. tau >0
    SigmaX : numpy array
        Covariance matrix of X
    SigmaY : numpy array
        Covariance matrix of Y
    divergence : string, optional
        RP or DPD. The default is "RP".
    previous_correlations : dict, optional
        Vector with preivous canonical vectors. The default is dict().
    init : numpy array, optional
        Initial vector for optimization algorithm. The default is False.
    Repetitions : int, optional
       Number of repetitions with different initializations. The default is 10.

    Returns
    -------
    estimates : numpy array
        vector with coefficients of canonical correlations

    """

    Xlen = X.shape[1]
    Ylen = Y.shape[1]
    
    if not len(previous_correlations):
        
         estimates = opt (X = X,
                           Y = Y,
                           tau = tau,
                           SigmaX = SigmaX,
                           SigmaY = SigmaY,
                           divergence = divergence,
                           previous_correlations = previous_correlations,
                           init = init,
                           Repetitions =  Repetitions 
                           )
         
    else:
        
        A=np.zeros(shape=(Xlen,len(previous_correlations)))
        B=np.zeros(shape=(Ylen,len(previous_correlations)))
        
        for cv in range(len(previous_correlations)):
            A[:,cv] = previous_correlations[str(cv)][:Xlen]
            B[:,cv] = previous_correlations[str(cv)][Xlen:]
        A = A @ A.T
        B= B @ B.T
        
        Au, _, _ = np.linalg.svd(A, full_matrices=True) #quiza As, Avh (otros resultados) sean necesarios en otro momento 
        Bu, _, _ = np.linalg.svd(B, full_matrices=True) #por ahora no los guardo
        
        Auast = Au[:,len(previous_correlations):]
        tXp = X.dot(Auast)
        Buast = Bu[:,len(previous_correlations):]
        tYp = Y.dot(Buast)
        
        if init is not False:
            inita = init[:Xlen].dot(Auast)
            initb = init[Xlen:].dot(Buast)
            init = np.concatenate((inita, initb))
        
        estimates = opt (X =tXp,
                         Y = tYp,
                         tau = tau,
                         SigmaX = np.identity(tXp.shape[1]),
                         SigmaY = np.identity(tYp.shape[1]),
                         divergence = divergence,
                         previous_correlations = dict(),
                         init = init,
                         Repetitions =  Repetitions 
                          )
        estimates = np.concatenate( (Auast.dot(estimates[:tXp.shape[1]]), Buast.dot(estimates[tXp.shape[1]:]) ), axis=0 )

    return estimates

def opt (X,Y, tau, SigmaX, SigmaY, divergence = "RP", previous_correlations = dict(), init=False, Repetitions = 10):
    
    """
    Optimization function. Return the optimized function.
    
    Parameters
    ----------
    X : numpy array
        Data group 1
    Y : numpy array
        Data group 2
    tau : float
        divergence tuning parameter. tau >0
    SigmaX : numpy array
        Covariance matrix of X
    SigmaY : numpy array
        Covariance matrix of Y
    divergence : string, optional
        RP or DPD. The default is "RP".
    previous_correlations : dict, optional
        Vector with preivous canonical vectors. The default is dict().
    init : numpy array, optional
        Initial vector for optimization algorithm. The default is False.
    Repetitions : int, optional
       Number of repetitions with different initializations. The default is 10.

    Returns
    -------
    result.x : numpy array
        vector with coefficients of canonical correlations

    """
    Xlen = X.shape[1]
    Ylen = Y.shape[1]
     
    #Unit variance
    nlc_var_a = NonlinearConstraint(lambda c: cov_can_a (c, Xlen, SigmaX,c) , 1, 1)
    nlc_var_b = NonlinearConstraint(lambda c: cov_can_b (c, Xlen, SigmaY,c) , 1, 1)
    cons = [nlc_var_a, nlc_var_b]
    
    #Incorrelation
    for prev in range(len(previous_correlations)):
        nlc_var_a = NonlinearConstraint(lambda c: cov_can_a (c, Xlen, SigmaX, previous_correlations[str(prev)]) , 0, 0)
        nlc_var_b = NonlinearConstraint(lambda c: cov_can_b (c, Xlen, SigmaY, previous_correlations[str(prev)]) , 0, 0)
        cons.append(nlc_var_a)
        cons.append(nlc_var_b)
        
    if (init is False):
        if not len(previous_correlations):
            inita = [1/np.sqrt(np.sum(SigmaX))]*Xlen
            initb = [1/np.sqrt(np.sum(SigmaY))]*Ylen
            init = inita + initb
        else:
            init = initial_guess(SigmaX, SigmaY, previous_correlations)
    
    result = optimize.minimize(fun = distance, x0 = init, args = (X,Y, tau, divergence), constraints = cons)
    
    #test different initial points
    min_fun = result.fun 
    for i in range(Repetitions):
        
        init = initial_guess(SigmaX, SigmaY, previous_correlations)
        result_temp = optimize.minimize(fun = distance, x0 = init, args = (X,Y, tau, divergence), constraints=cons)
        min_temp = result_temp.fun 
        print(f'Iteration {i}: divergence = {min_temp}')
        
        if (min_temp < min_fun):
            result = result_temp
            min_fun = min_temp 
            
        #_early Stopping
        ace_xy = ace(np.stack((np.dot(Y,result_temp.x[Xlen:]), np.dot(X,result_temp.x[:Xlen]) ) ))
        #print(f'ACE at iteration {i}: ACE = {ace_xy}')
        
        if ace_xy > 0.95 :
            break
                
    print(f'Optimal value of the divergence = {min_fun}')
    #print(f'el error final es {get_metrics_transformed(real,Xcont, Ycont, sqrt_SigmaXcont, sqrt_SigmaYcont, result.x)}')
            
    return result.x




# def distance_permutation(permutation, real, tau,  divergence, previous_correlations = dict()):
    
#     Yp = real["Y"][permutation,:]
#     SigmaYp = np.cov(Yp, rowvar=False, ddof=0)
#     result=opt(real["X"],Yp,tau, real["SigmaX"], SigmaYp, divergence, previous_correlations)
#     fitted_params = result.x
#     dist = distance(fitted_params,real["X"],Yp,tau, divergence )
#     return(dist)


def distance_permutation(permutation, real, tau,  divergence, previous_correlations = dict()):
    """
    Distance computed for permutation test

    Parameters
    ----------
    permutation : TYPE
        DESCRIPTION.
    real : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    divergence : TYPE
        DESCRIPTION.
    previous_correlations : TYPE, optional
        DESCRIPTION. The default is dict().

    Returns
    -------
    None.

    """
    
    Yp = real["Y"][permutation,:]
    SigmaYp = np.cov(Yp, rowvar=False, ddof=0)
    result=opt(real["X"],Yp,tau, real["SigmaX"], SigmaYp, divergence, previous_correlations)
    fitted_params = result.x
    dist = distance(fitted_params,real["X"],Yp,tau, divergence )
    return(dist)

def contraste (real, tau, divergence, previous_correlations, nper=30):
    
    nobs = real["X"].shape[0]
    reference_distance = distance_permutation(np.arange(nobs), real, tau, divergence, previous_correlations)
    
    permutations = np.array(list(map(lambda x: random.sample(range(nobs), nobs), range(nper))))
    permuted_distances = np.apply_along_axis(lambda x: distance_permutation(x, real, tau, divergence, previous_correlations), 1, permutations)
    
    p_value = np.mean(np.abs(permuted_distances) > np.abs(reference_distance))

    return p_value