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
from codes.kernels_for_GPK import *
import codes.config

from ipywidgets import IntProgress

# Aug 2020 Mengyan Zhang
# Implement batch ucb, with three methods:
# 1. Return Top n arm directly (non-batch)
# 2. Clustering 
# 3. GP-BUCB

KERNEL_DICT = {
    # 'Spectrum_Kernel': Spectrum_Kernel,
    # 'WD_Kernel': WeightedDegree_Kernel,
    # 'Sum_Spectrum_Kernel': Sum_Spectrum_Kernel,
    # 'Mixed_Spectrum_Kernel': Mixed_Spectrum_Kernel,
    'WD_Kernel_Shift': WD_Shift_Kernel   
}


class RBS_UCB():
    """GPUCB for RBS sequences design.

    Attributes
    -----------------------------------------------
    df_known: pandas dataframe
        known RBS sequences with labels
        train data
    df_design: pandas dataframe
        core part design (4^6) expect those already in df_known
        test data
    df_train_test: pandas dataframe
        concat of df_known and df_design

    kernel_name: string
        indicates kernel to use 
    l: int
        lmer
    s: int
        shift number
    embedding: string
        label or onehot
    alpha: float
        parameter for GPR, value added to kernel diagonal
    
    rec_size: int
        recommendation size
    beta: float
        parameter for UCB, mean + beta * std
        balance exploration and exploitation
    
    """
    def __init__(self, df_known, kernel_name='WD_Kernel_Shift', l=6, s=1, sigma_0 = 1,
                embedding='label', alpha=2, rec_size=90, beta=2, kernel_norm_flag = True,
                centering_flag = True, unit_norm_flag = True, kernel_over_all_flag = True,
                df_design = None):
        self.df_known = df_known
        self.df_known['train_test'] = 'Train'
        self.known_rbs_set = set(self.df_known['RBS'])
        # if df_design is None:
        #     self.df_design = self.generate_design_space()
        # else:
        #     self.df_design = df_design
        self.df_design = self.generate_design_space(df_design)
        self.df_design['train_test'] = 'Test'
        self.df_train_test = pd.concat([self.df_known, self.df_design], sort = True) #.reset_index()
        self.df_train_test = self.df_train_test.set_index('idx')

        # initialization for regression
        self.kernel_name = kernel_name
        self.l = l
        self.s = 1
        self.sigma_0 = sigma_0 
        
        self.embedding = embedding
        self.alpha = alpha

        self.gpr = GPR_Predictor(
                self.df_train_test, 
                train_idx = self.df_train_test['train_test'] == 'Train', 
                test_idx = self.df_train_test['train_test'] == 'Test', 
                kernel_name = self.kernel_name, 
                alpha=self.alpha, 
                embedding=self.embedding,
                l=self.l, 
                s = self.s,
                sigma_0= self.sigma_0,
                eva_on = 'seq', # for design
                kernel_norm_flag= kernel_norm_flag,
                centering_flag=centering_flag,
                unit_norm_flag=unit_norm_flag,
                kernel_over_all_flag = kernel_over_all_flag
                )

        # initialization for ucb parameters
        self.rec_size = rec_size 
        self.beta = beta # ucb =  mean + beta * std
        # TODO: the choice of beta is critical

    def generate_design_space(self, df_design):
        # create all combos

        if df_design is None:
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
       
            df_design['RBS'] = [x for x in combos if x not in self.known_rbs_set]
        
        df_design['RBS6'] = df_design['RBS'].str[7:13]
        df_design['idx'] = None
        
        with open(config.SAVED_IDX_SEQ_PATH, 'rb') as handle:
            idx_seq = pickle.load(handle)

        idx_seq_dict = idx_seq['idx_seq_dict']

        for i, rbs in df_design.iterrows():
            df_design.loc[i, 'idx'] = idx_seq_dict[rbs['RBS']]

        return df_design

    def prediction(self):
        # use Gaussian Process Regression
     
        self.gpr.regression()

        # update with pred mean and std
        self.df_known = self.gpr.train_df
        self.df_design = self.gpr.test_df[['RBS', 'RBS6', 'AVERAGE', 'pred mean', 'pred std']].copy()

        # add ucb and lcb
        self.df_design['ucb'] = self.df_design['pred mean'] + self.beta * self.df_design['pred std']
        self.df_design['lcb'] = self.df_design['pred mean'] - self.beta * self.df_design['pred std']

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
        if self.rec_size == None:
            return self.df_design.sort_values(by = 'ucb', ascending=False)
        else:
            return self.df_design.sort_values(by = 'ucb', ascending=False).head(self.rec_size)


class Batch_clustering_ucb(RBS_UCB):
    """Clustering idea.
    # TODO: test
    """
    def __init__(self, df_known, kernel_name='WD_Kernel_Shift', l=6, s=1,
                embedding='label', alpha=0.1, rec_size=90, beta=1, n_clusters=90):
        self.n_clusters = n_clusters
        super().__init__(self, df_known, kernel_name, l, s,
                embedding, alpha, rec_size, beta)

    def recommendation(self):
        distance = self.gpr.kernel_instance.distance_all
        k_medoids = KMedoids(n_clusters=self.n_clusters,
                            metric = 'precomputed',
                            init='k-medoids').fit(distance)
        y_km_spec = kmedoids.labels_
        self.df_train_test['cluster'] = y_km_spec

        max_ucb_in_clusters = pd.DataFrame(columns=['RBS', 'ucb', 'pred mean', 'pred std', 'lcb'])

        for group, value in sorted_ucb.groupby('cluster'):
            max_ucb_in_clusters.loc[group] = value.sort_values('ucb', ascending = False)[['RBS', 'ucb', 'pred mean', 'pred std', 'lcb']].iloc[0]

        sorted_max_ucb_in_clusters = max_ucb_in_clusters.sort_values('ucb', ascending=False) 

        return sorted_max_ucb_in_clusters[: self.rec_size]


class GP_BUCB(RBS_UCB):
    """
    Desautels et al. 2014 Algorithm 2
    http://jmlr.org/papers/volume15/desautels14a/desautels14a.pdf
    """

    def run_experiment(self):
        # TODO: make it online prediction, the key point is to handle kernel matrix

        rec_df = pd.DataFrame()

        self.prediction()
        batch_df = self.df_design.copy()

        for i in range(self.rec_size):
            
            sorted_ucb_batch = batch_df.sort_values(by = 'ucb', ascending=False)
            # print('sorted_batch_df')
            # print(sorted_ucb_batch.head(5))
            rec = sorted_ucb_batch.head(1)
            
            # rec_df = rec_df.append(rec, ignore_index =True)
            rec_df = rec_df.append(rec)

            rec_idx = sorted_ucb_batch.index[0]
            # print('rec index ', rec_idx)
            self.df_train_test.loc[rec_idx, 'train_test'] = 'Train'
            print('train size ', self.df_train_test[self.df_train_test['train_test'] == 'Train'].shape)

            # add replicates label to avoid being droped
            # rec_rbs = sorted_ucb_batch['RBS'].values[0]
            # all_rep_rec = self.gpr.df['RBS'] == rec_rbs
            # print(self.gpr.df.loc[all_rep_rec])
            # self.gpr.df.loc[all_rep_rec,'Rep2'] = self.gpr.test_df.loc[rec_idx,'pred mean']
            # self.gpr.df.loc[all_rep_rec,'AVERAGE'] = self.gpr.test_df.loc[rec_idx,'pred mean']
            
            # TODO: one sequence only has one replicate in testing data; 
            # but in training data, one sequence has 6 replicates
            self.gpr.df.loc[rec_idx,'Rep2'] = self.gpr.test_df.loc[rec_idx,'pred mean']
            self.gpr.df.loc[rec_idx,'AVERAGE'] = self.gpr.test_df.loc[rec_idx,'pred mean']
            # print(self.gpr.df.loc[rec_idx, 'RBS'])
            # print(self.gpr.test_df.loc[rec_idx, 'RBS'])
            
            # update train test idx
            self.gpr.train_idx = self.df_train_test['train_test'] == 'Train', 
            self.gpr.test_idx = self.df_train_test['train_test'] == 'Test', 
            self.gpr.regression()

            # use unchanged mean, updated std
            batch_df = self.gpr.test_df
            batch_df.loc[:, 'pred mean'] = self.df_design.loc[np.asarray(batch_df.index), 'pred mean']
            batch_df['ucb'] = batch_df['pred mean'] + self.beta * batch_df['pred std']
            batch_df['lcb'] = batch_df['pred mean'] - self.beta * batch_df['pred std']
            
         
        return rec_df[['RBS', 'RBS6', 'AVERAGE', 'pred mean', 'pred std', 'ucb', 'lcb']].copy()


