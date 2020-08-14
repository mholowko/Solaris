# Implement batch ucb, with three methods:
# 1. Return Top n arm directly (non-batch)
# 2. Clustering 
# 3. GP-BUCB

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

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel, DotProduct, RBF 
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import KFold
from sklearn_extra.cluster import KMedoids

from codes.embedding import Embedding
from codes.environment import Rewards_env
from codes.ucb import GPUCB, Random
from codes.evaluations import evaluate, plot_eva
from codes.regression import *
from codes.kernels_for_GPK import Spectrum_Kernel, Sum_Spectrum_Kernel, WeightedDegree_Kernel

from ipywidgets import IntProgress

KERNEL_DICT = {
    'Spectrum_Kernel': Spectrum_Kernel,
    'WD_Kernel': WeightedDegree_Kernel,
    'Sum_Spectrum_Kernel': Sum_Spectrum_Kernel,
    'Mixed_Spectrum_Kernel': Mixed_Spectrum_Kernel,
    'WD_Kernel_Shift': WD_Shift_Kernel
    
}


class RBS_UCB():
    def __init__(self, known_df, kernel_name='WD_Kernel_Shift', 
                normalise_kernel_flag='True', embedding='label', alpha=0.1,
                eva_metric=mean_squared_error, l_list=[6], s=1,
                rec_size=90, beta=1):
        self.known_df = known_df
        self.known_df['train_test'] = 'Train'
        self.known_rbs_set = set(self.known_df['RBS'])
        self.df_design = self.generate_design_space()
        self.df_design['train_test'] = 'Test'
        self.df_train_test = pd.concat([self.known_df, self.df_design], sort = True).reset_index()

        # initialization for regression
        self.kernel_name = kernel_name
        self.normalise_kernel_flag = normalise_kernel_flag
        self.embedding = embedding
        self.alpha = alpha
        self.eva_metric = eva_metric 
        self.l_list = l_list
        self.s = 1

        # initialization for ucb parameters
        self.rec_size = rec_size 
        self.beta = beta # ucb =  mean + beta * std
        # TODO: the choice of beta is critical

    def generate_design_space(self):
        # create all combos

        combos = [] # 20-base
        combos_6 = [] # 6-base

        # Setting
        char_sets = ['A', 'G', 'C', 'T']
        design_len = 6
        pre_design = 'TTTAAGA'
        pos_design = 'TATACAT'

        for combo in itertools.product(char_sets, repeat= design_len):
            combo = pre_design + ''.join(combo) + pos_design
            combos_6.append(''.join(combo))
            combos.append(combo)
            
        assert len(combos) == len(char_sets) ** design_len

        df_design = pd.DataFrame()
        df_design['RBS'] = list(set(combos) - self.known_rbs_set)

        return df_design

    def prediction(self):
        # use Gaussian Process Regression
        self.gpr = GPR_Predictor(
                        self.df_train_test, 
                        train_idx = self.df_train_test['train_test'] == 'Train', 
                        test_idx = self.df_train_test['train_test'] == 'Test', 
                        kernel_name = self.kernel_name, 
                        normalise_kernel = self.normalise_kernel_flag, 
                        alpha=self.alpha, 
                        embedding=self.embedding,
                        eva_metric=self.eva_metric, 
                        l_list=self.l_list, 
                        s = self.s
                        )

        self.gpr.regression()

        # update with pred mean and std
        self.known_df = self.gpr.train_df
        self.df_design = self.gpr.test_df

        # add ucb and lcb
        self.df_design['ucb'] = self.df_design['pred mean'] + \
                                self.beta * self.df_design['pred std']
        self.df_design['ucb'] = self.df_design['pred mean'] + \
                                self.beta * self.df_design['pred std']

        

    def recommendation(self):
        """Recommendation
        """

    def run_experiment(self):
        self.prediction()
        rec_df = self.recommendation()
        
        return rec_df

class Top_n_ucb(RBS_UCB):
    def recommendation(self):
        """Recommendation
        """
        return self.df_design.sort_values(by = 'ucb', ascending=False)[:self.rec_size]


class Batch_clustering_ucb(RBS_UCB):
    """Clustering idea.
    """


class GP_BUCB(RBS_UCB):
    """
    Desautels et al. 2014 Algorithm 2
    http://jmlr.org/papers/volume15/desautels14a/desautels14a.pdf
    """

    def run_experiment(self):
        rec_df = pd.DataFrame()

        for i in range(self.rec_size):
            self.prediction()
            sorted_ucb_batch = self.df_design.sort_values(by = 'ucb', ascending=False)
            rec = sorted_ucb_batch.head(1)
            rec_df = rec_df.append(rec, ignore_index =True)

            rec_idx = sorted_ucb_batch.index[0]
            self.df_train_test['train_test'] = 'Train'

            # add replicates label to avoid being droped
            self.gpr.df.loc[rec_idx,'Rep2'] = self.gpr.test_df.loc[rec_idx,'pred mean']
            self.gpr.df.loc[rec_idx,'AVERAGE'] = self.gpr.test_df.loc[rec_idx,'pred mean']

        return rec_df



