# Simulate the bandits recommendations on lit preds
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
result_path = '../data/Results_Lit.csv'
whole_size = 400

if not os.path.exists(result_path):
    data_path = '../data/Comparison_Data/Lit_Data.csv'
    data_df = pd.read_csv(data_path, header= 0)
    plt.hist(data_df['Label'])
    plt.savefig('Label_hist.png')

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

    log_lit = np.log(data_df['Label'])
    data_df['Label_zscoreNorm'] = (log_lit - log_lit.mean())/log_lit.std()
    print(data_df.sort_values(by = ['Label_zscoreNorm']))
    print('Label_zscoreNorm mean: ', data_df['Label_zscoreNorm'].mean())
    print('Label_zscoreNorm std: ', data_df['Label_zscoreNorm'].std())
    data_df['RBS_combine'] = data_df['RBS_n'].str[11:21] + data_df['RBS_m'].str[10:20]
    # data_df['RBS_combine'] = data_df['RBS_combine'].str[10:-5]
    print(len(data_df.loc[0, 'RBS_combine']))
    data_df['RBS_Core'] = data_df['RBS_n'].str[12:14] + data_df['RBS_n'].str[18:20] + data_df['RBS_m'].str[11:14] + data_df['RBS_m'].str[16:19]

    plt.figure()
    plt.hist(data_df['Label_zscoreNorm'])
    plt.savefig('log_Label_zscoreNorm_hist.png')
    print(data_df[['RBS_combine', 'RBS_Core', 'Label_zscoreNorm']].head())

    df = pd.DataFrame(columns=['Name','Group','Plate','Round','RBS','RBS6','Rep1','Rep2', 'Rep3', 'Rep4','Rep5','Rep6','AVERAGE', 'STD'])
    df[['RBS', 'RBS6', 'AVERAGE']] = data_df[['RBS_combine', 'RBS_Core','Label_zscoreNorm']]
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

rec_size = 20 # in each round, we recommend 90 RBS sequences
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

n_repeat = 2
total_round = 5

all_recs = []
# save_folder_path = './sim_results'
save_folder_path = '/home/v-mezhang/blob/Solaris/sim_results/lit/topucb'
print('save folder path: ', save_folder_path)
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

idx_seq_dict = {}
index_list = []
seq_list = []
for i in range(whole_size):
    index_list.append(i)
    seq_list.append(df.loc[i, 'RBS'])
    idx_seq_dict[df.loc[i, 'RBS']] = i
    df.loc[i, 'idx'] = i
# print(idx_seq_dict)
save_dict = {}
save_dict['idx_seq_dict'] = idx_seq_dict
save_dict['idx_list'] = index_list
save_dict['seq_list'] = seq_list
with open('../data/idx_seq_lit.pickle', 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(df)

for i in range(1,n_repeat):
    rec_dfs = []

    df['Round'] = total_round
    recs = []
    for design_round in range(total_round):
        # UCB hyperparameter
        if design_round == total_round - 1: # last round
            beta = 0
        else:
            beta = 2

        # Debug: the bandit rec finds the best ones at the beginning, but with decreased performance. I guess the reason is we set beta = 2, for exploration. Let me try to set beta = 0 for bandit-i (where I >= 1)
        # if design_round <=1:
        #     beta = 2
        # else:
        #     beta = 0

        if design_round == 0:
            # random sample
            print('Design round {}: randomly generate {} recommendations.'.format(design_round, rec_size))

            rec_idxs = np.random.choice(range(whole_size), size = rec_size, replace=False)
            # rec first rec size
            # rec_idxs = range((0 + i) * rec_size, (1+i)* rec_size)
            # print(rec_idxs)
            rec_rbs_list = [df.loc[i, 'RBS'] for i in rec_idxs]
            rec_rbs6_list = [df.loc[i, 'RBS'][7:13] for i in rec_idxs]

            # rec worst averages
            # rec_rbs_list = list(df.sort_values(by = ['AVERAGE'])[:rec_size]['RBS'])
            # rec_rbs6_list = list(df.sort_values(by = ['AVERAGE'])[:rec_size]['RBS6'])

            # print(rec_rbs_list)
            rec_df = pd.DataFrame(columns=['idx', 'RBS','RBS6','AVERAGE', 'pred mean', 'pred std', 'ucb', 'lcb', 'Group'])
            rec_df['RBS'] = rec_rbs_list
            rec_df['RBS6']  = rec_rbs6_list
            rec_df['Group'] = 'Random'
            for j, rbs in rec_df.iterrows():
                rec_df.loc[j, 'idx'] = idx_seq_dict[rbs['RBS']]
            rec_df = rec_df.set_index('idx') 
        else:
            print('Design round {}: '.format(design_round))
            # Top_n_ucb, GP_BUCB
            gpbucb = Top_n_ucb(df[df['Round'] < design_round], kernel_name=kernel, l=l, s=s,      
                            sigma_0=sigma_0,
                            embedding=embedding, alpha=alpha, rec_size=rec_size, beta=beta, 
                            kernel_norm_flag=kernel_norm_flag, centering_flag = centering_flag,              
                            unit_norm_flag=unit_norm_flag, kernel_over_all_flag = kernel_over_all_flag, 
                            df_design= df[df['Round'] >= design_round])

            rec_df = gpbucb.run_experiment()
            rec_df['Group'] = 'Bandit-' + str(design_round-1)
            rec_rbs_list = list(rec_df['RBS'])

        for j in range(whole_size):
            if df.loc[j, 'RBS'] in rec_rbs_list:
                df.loc[j, 'Round'] = design_round 

        rec_df['Round'] = design_round
        print(rec_df)
        rec_dfs.append(rec_df)
        recs.append(rec_rbs_list)
    rec_dfs = pd.concat(rec_dfs) # , ignore_index=True
    rec_dfs = rec_dfs.merge(df[['RBS', 'AVERAGE']], how='left', on = 'RBS')
    rec_dfs = rec_dfs.rename(columns = {'AVERAGE_y': 'AVERAGE'}).drop(columns = ['AVERAGE_x'])
    print(rec_dfs)
    rec_dfs.to_csv(os.path.join(save_folder_path, 'recs_' + str(i) + '_' + str(rec_size) + '.csv'))
    print('Save rec dfs to ', os.path.join(save_folder_path, 'recs_' + str(i) + '_' + str(rec_size) + '.csv'))
    all_recs.append(recs)

    # np.save('all_recs_'+ str(i) + '_' + str(rec_size) + '.npy', all_recs)