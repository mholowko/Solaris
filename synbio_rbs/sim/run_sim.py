# Simulate the bandits recommendations on Salis preds
# 
# The pipeline includes the following steps:
# - data pre-processing
# - prediction (GPR)
# - batch UCB recommendation
# 

# %%
# direct to proper path
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

import itertools
from collections import defaultdict
import math
import json
# import xarray as xr

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel, DotProduct, RBF 
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE

from src.embedding import Embedding
from src.environment import Rewards_env
from src.evaluations import evaluate, plot_eva
from src.regression import *
from src.kernels_for_GPK import *
from src.data_generating import generate_data
from src.batch_ucb import *
import src.config
import warnings
from platform import python_version
print(python_version())

# folder_path = '../data/'
# raw = 'n'
# create df in required form
result_path = '../data/Results_Salis.csv'
whole_size = 4096

if not os.path.exists(result_path):
    data_path = '../data/Comparison_Data/Model_Comparisons.csv'
    data_df = pd.read_csv(data_path, header= 0)

    # # Data pre-processing
    # 
    # Our data pre-processing involved the following options:  
    # - a. In each round, substract the mean of every data points by the reference AVERAGE, and then add 100 (to make the values positive).  
    # - b. Take log (base e) transformation for each data point.  
    # - c. Apply z-score normalisation.  
    #     - c.1 on each round, so that the mean and variance of each replicate of data in each round is zero and one after normalisation. 
    #     - c.2 on all data, so that the mean and variance of each replicate of all data is zero and one after normalisation. 
    # - d. Apply min-max normalisation.
    #     - d.1 on each round
    #     - d.2 on all data
    # - e. Apply ratio normalisation. In each round, each data points is devided by the mean of refernce AVERAGE, so that in each round, the reference labels are almost 1. 
    #     - e.1 on each round
    #     - e.2 on all data
    #     
    # In Round 1 (Bandit-1), we adopt *bc1*. We have observed that the reference sequences give differerent TIR values in each round. Thus in Round 2-3 (Bandit-3), we substructed the mean first and adopted *abc1*.
    # 
    # 
    # The source code of data generating approaches is defined in src/data_generating.py.

    # round1='bc1'
    # round23 = 'abc1'

    # corresponding to bc2 (since there is round, do not need to run a, and replace c1 to c2)

    log_salis = np.log(data_df['TIR_Salis'])
    data_df['TIR_Salis_zscoreNorm'] = (log_salis - log_salis.mean())/log_salis.std()
    print(data_df.sort_values(by = ['TIR_Salis_zscoreNorm']))
    print('TIR_Salis_zscoreNorm mean: ', data_df['TIR_Salis_zscoreNorm'].mean())
    print('TIR_Salis_zscoreNorm std: ', data_df['TIR_Salis_zscoreNorm'].std())

    # plt.hist(df['TIR_Salis_zscoreNorm'])
    # plt.savefig('log_TIR_Salis_zscoreNorm_hist.png')


    df = pd.DataFrame(columns=['Name','Group','Plate','Round','RBS','RBS6','Rep1','Rep2', 'Rep3', 'Rep4','Rep5','Rep6','AVERAGE', 'STD'])
    df[['RBS', 'RBS6', 'AVERAGE']] = data_df[['RBS_sequence', 'RBS_Core','TIR_Salis_zscoreNorm']]
    for i in range(whole_size):
        df.loc[i,'Name'] = i
        df.loc[i,'Round'] = 10 # bigger than design round
        df.loc[i,'STD'] = 0.3 # average std in previous data
        for j in range(1, 7):
            df.loc[i, 'Rep' + str(j)] = np.random.normal(df.loc[i, 'AVERAGE'], df.loc[i, 'STD'])

    df.to_csv(result_path, index=False)
    print('Save result file to {}'.format(result_path))
else:
    print('Load result file from {}'.format(result_path))
    df = pd.read_csv(result_path, header = 0)

# ## Prediction & Recommendation
# The prediction code is mainly located in *src.regression.py*, the recommendation code is in $src.batch\_ucb.py$.
# To predict, the *src.kernels\_for\_GPK.py* is called to calculate the kernel functions.
# When we call the GP_BUCB function, the prediction (GPR) is firstly called and recommendation is conducted based on the recommendation. 
# 
# The settings are specified as in the next cell. To generate the recommendation result for round n, change the parameter *design_round = n*.

# setting

rec_size = 90 # in each round, we recommend 90 RBS sequences
l = 6 # maximum kmer as 6
s = 1 # maximum shift as 1
alpha = 2 # GPR noise parameter, get from cross validation
sigma_0 = 1 # signal for kernel matrix 
kernel = 'WD_Kernel_Shift' # weighted degree kernel with shift
embedding = 'label' # turns strings into categories first and used for kernel 
kernel_norm_flag = True # whether to apply kernel normalisation
centering_flag = True # whether to apply kernel centering
unit_norm_flag = True # whether to apply unit norm for kernel
kernel_over_all_flag = True  # we keep the setting the same as our last round
# if design_round == 1:  # kernel normalisation over what
#     kernel_over_all_flag = False
#     df = df_round1
# else:
#     kernel_over_all_flag = True
#     df = df_round23

n_repeat = 3
total_round = 4

all_recs = []

for i in range(n_repeat):
    df['Round'] = total_round
    recs = []
    for design_round in range(total_round):
        # UCB hyperparameter
        if design_round == total_round - 1: # last round
            beta = 0
        else:
            beta = 2

        if design_round == 0:
            # random sample
            print('Design round {}: randomly generate {} recommendations.'.format(design_round, rec_size))
            rec_idxs = np.random.choice(range(whole_size), size = rec_size, replace=False)
            # print(rec_idxs)
            rec_rbs_list = [df.loc[i, 'RBS'] for i in rec_idxs]
        else:
            print('Design round {}: '.format(design_round))
            gpbucb = Top_n_ucb(df[df['Round'] < design_round], kernel_name=kernel, l=l, s=s,      
                            sigma_0=sigma_0,
                            embedding=embedding, alpha=alpha, rec_size=rec_size, beta=beta, 
                            kernel_norm_flag=kernel_norm_flag, centering_flag = centering_flag,              
                            unit_norm_flag=unit_norm_flag, kernel_over_all_flag = kernel_over_all_flag)

            gpbucb_rec_df = gpbucb.run_experiment()
            rec_rbs_list = list(gpbucb_rec_df['RBS'])

        for i in range(whole_size):
            if df.loc[i, 'RBS'] in rec_rbs_list:
                df.loc[i, 'Round'] = design_round 

        recs.append(rec_rbs_list)
    all_recs.append(recs)

    np.save('all_recs_topnUCB.npy', all_recs)