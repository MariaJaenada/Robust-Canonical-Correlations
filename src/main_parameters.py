# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:41:28 2022

@author: mjaenada
"""
import pandas as pd

from generate_matrices_functions import generate_matrices3, generate_matrices4, generate_matrices5
from  simulation_transformed import simulate, loopfun_robust

N = 100
TAU_LIST = [0,0.2,0.4,0.6]

GENERATE_MATRICES_FUNCTION = generate_matrices5
DIVERGENCE = "DPD"
CONTAMINATION = 0.15
LOOPFUN = loopfun_robust
MAX_CORRELATIONS = 2
R=10
PERMUTATION_TEST = False

it, res= simulate(n=N,
                  tau_list=TAU_LIST, 
                  generate_matrices_function = GENERATE_MATRICES_FUNCTION,
                  divergence = DIVERGENCE,
                  contamination = CONTAMINATION,
                  loopfun= LOOPFUN,
                  max_correlations=MAX_CORRELATIONS,
                  R=R, 
                  permutation_test = PERMUTATION_TEST
                  )
res
pd.concat(res).to_csv("DPD01.csv")
  

