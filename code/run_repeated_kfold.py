
# [Update Aug 2020 Mengyan Zhang]
# 
# This notebook implements regression on the first round results with repeated kfold.
# 
# # Pipeline
# 
# - Data pre-processing: run codes/data_generating.py
#     - log transform 
#     - z-score normalisation for each replicate (zero mean and unit variance)
# - Repeated Kfold: (n_repeated: 10; n_split/kfold: 5)
#     - Kernel: weighted degree kernel with shift
#         - kernel normalisation: centering and unit norm
#         - lmer: number of substring [3,4,5,6]
#     - Gaussian process regression 
#         - alpha: scalar value add to diagonal 
# - Evaluation
#    - metric: e.g. Mean square error; R2
#    - true label: either sample or mean of sample. 
# # Key Notes
# 
# ## Splitting over sequences
# 
# The training and testing data should be split in terms of sequences rather than samples, since we hope to have good predictions on unseen data. Similar idea as shown in [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
# 
# ## Training with multiple labels vs. sample mean?
# 
# We now training with multiple labels, i.e. repeated sequences inputs with different labels (replicates).
# It worth thinking whether it is equivalent to use the sample mean directly.
# 
# 
# ## Evaluate on samples vs sample mean?
# 
# For evaluation (on both training and testing predictions), we evaluate using "samples" or "averages", indicating by "eva_on" parameter. 
# 
# ## What matters
# 
# The recommendations at the end it what matters, so once we choose certain parameters, we should focus on how it changes our recommendations.

# FIXME: fix path
# direct to proper path
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from collections import defaultdict
import math
import json
import xarray as xr
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel, DotProduct, RBF 
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import KFold

from codes.embedding import Embedding
from codes.environment import Rewards_env
from codes.ucb import GPUCB, Random
from codes.evaluations import evaluate, plot_eva
from codes.regression import *
from codes.kernels_for_GPK import *

from ipywidgets import IntProgress
from IPython.display import display
import warnings

kernel_dict = {
#     'Spectrum_Kernel': Spectrum_Kernel,
#     'Mixed_Spectrum_Kernel': Mixed_Spectrum_Kernel,
#     'WD_Kernel': WeightedDegree_Kernel,
#     'Sum_Spectrum_Kernel': Sum_Spectrum_Kernel,
    'WD_Kernel_Shift': WD_Shift_Kernel,
    'RBF': RBF
    
}

Path = '../../data/Results_Microplate_partialTrue_normTrue_roundRep_formatSeq_logTrue.csv'
# save_path = 'repeated_kfold_RBF_round01_kenrelNormUniqueAdjust.pickle'
save_path = '../../data/repeated_kfold_test.pickle'
df = pd.read_csv(Path)

# # Repeated KFold 

kernel = 'WD_Kernel_Shift'
eva_metric = [mean_squared_error, r2_score, 'coverage rate']

gpr = GPR_Predictor(df, kernel_name = kernel)
# gpr = GPR_Predictor(df, train_idx=df['Round'] == 0, test_idx=df['Round'] == 1, kernel_name = kernel)

# num_split = 5
# num_repeat = 5
# s_list = [0,1]
# # alpha_list = [0.5]
# alpha_list= [1e-5, 1e-1, 0.5, 1, 2, 5] 
# # alpha_list= [0.01, 0.05, 0.1, 0.5].append(list(range(1,15)))
# l_list =[3,6]
# # sigma_0_list = [0.5, 1, 1.5, 2, 2.5]
# sigma_0_list = [0.5, 1, 2]

num_split = 5
num_repeat = 10
s_list = [1]
alpha_list = [2]
# alpha_list= [1e-5, 1e-1, 0.5, 1, 2, 5] 
# alpha_list= [0.01, 0.05, 0.1, 0.5].append(list(range(1,15)))
l_list =[6]
# sigma_0_list = [0.5, 1, 1.5, 2, 2.5]
sigma_0_list = [1]
                                                                                                         
result_DataArray = gpr.Repeated_kfold(num_split=num_split, num_repeat=num_repeat,
                                      kernel_norm_flag=[True], centering_flag=[True, False], unit_norm_flag=[True, False],
                                      alpha_list= alpha_list, l_list = l_list, s_list = s_list, sigma_0_list = sigma_0_list)


with open(save_path, 'wb') as handle:
    pickle.dump(result_DataArray, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(save_path, 'rb') as handle:
#     result_pkl = pickle.load(handle)
# result_pkl.loc[dict(train_test = 'Test')].loc[dict(eva_metric='coverage rate')]
# # result_pkl[1][1][1].loc[dict(centering_flag = True)].loc[dict(unit_norm_flag = False)].mean(axis = -1).mean(axis = -1)
# result_pkl[1][1][1].mean(axis = -1).std(axis = -1)
# # result_pkl[1][1][2].loc[dict(s = 1)].loc[dict(l=6)].mean(axis = -1).mean(axis = -1).plot()
# result_pkl[1].mean(axis = -1).mean(axis = -1).plot()
