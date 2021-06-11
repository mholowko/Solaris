import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import seaborn as sns
import itertools
from collections import defaultdict
import math
import json
import xarray as xr

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel, DotProduct, RBF 
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import KFold

from codes.embedding import Embedding
from codes.environment import Rewards_env
from codes.ucb import GPUCB, Random
from codes.evaluations import evaluate, plot_eva
#from codes.regression import Regression
from codes.kernels_for_GPK import *

kernel_dict = {
    'Spectrum_Kernel': Spectrum_Kernel,
    'Mixed_Spectrum_Kernel': Mixed_Spectrum_Kernel,
    'WD_Kernel': WeightedDegree_Kernel,
    'Sum_Spectrum_Kernel': Sum_Spectrum_Kernel,
    'WD_Kernel_Shift': WD_Shift_Kernel
    
}

def Train_test_split(num_data, test_size = 0.2, random_state = 24):
    np.random.seed(random_state)
    test_idx = np.random.choice(num_data, int(test_size * num_data), replace=False)
    train_idx = list(set(range(num_data)) - set(test_idx))
    
    return np.asarray(train_idx), np.asarray(test_idx)

def Train_val_split(num_data, cv = 5, random_state = 24):
    kf = KFold(n_splits = cv, shuffle = True)
    return kf.split(range(num_data))

def Generate_train_test_data(df, train_idx, test_idx, embedding):
    if 'Rep1' in df.columns:
        train_df = pd.melt(df.loc[train_idx], id_vars=['RBS', 'RBS6', 'AVERAGE', 'STD', 'Group'], value_vars=['Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5'])
        train_df = train_df.dropna()
        train_df = train_df.rename(columns = {'value': 'label'})
        test_df = pd.melt(df.loc[test_idx], id_vars=['RBS', 'RBS6', 'AVERAGE', 'STD', 'Group'], value_vars=['Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5'])
        test_df = test_df.dropna()
        test_df = test_df.rename(columns = {'value': 'label'})
    else: 
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]
        
    X_train = Rewards_env(np.asarray(train_df[['RBS', 'label']]), embedding).embedded
    y_train_sample = np.asarray(train_df['label'])
    y_train_ave = np.asarray(train_df['AVERAGE'])
    if 'STD' in train_df.columns:
        y_train_std = np.asarray(train_df['STD'])
    else:
        y_train_std = None
    
    X_test = Rewards_env(np.asarray(test_df[['RBS', 'label']]), embedding).embedded
    y_test_sample = np.asarray(test_df['label'])
    y_test_ave = np.asarray(test_df['AVERAGE']) 
    if 'STD' in test_df.columns:
        y_test_std = np.asarray(test_df['STD'])
    else:
        y_test_std = None
    
    return train_df, test_df, X_train, X_test, y_train_sample, y_test_sample, y_train_ave, y_test_ave, y_train_std, y_test_std


def cross_val(self, df, cv = 5, random_state = 24, test_size = 0.2, kernel_list = ['Spectrum_Kernel', 'Sum_Spectrum_Kernel'],
            alpha_list = [0.1, 1], embedding = 'label', eva_metric = r2_score, eva_on_ave_flag = True,
            l_lists = [[3]], b_list = [0.33], weight_flag = False, padding_flag = False, gap_flag = False):
    """Regression with double cross validation, finding best hyperparameters on the validation set. 
    """
    test_scores = []
    
    data = np.asarray(df[['RBS', 'AVERAGE']])
    num_data = data.shape[0]
    
    # train_idx, test_idx = Train_test_split(num_data, test_size, random_state)
    # train_df, test_df, X_train, X_test, y_train_sample, y_test_sample, y_train_ave, y_test_ave, y_train_std, y_test_std\
    #   = Generate_train_test_data(df, train_idx, test_idx, embedding)

    for train_idx, test_idx in Train_val_split(num_data, cv = cv, random_state = random_state):
        train_df, test_df, X_train, X_test, y_train_sample, y_test_sample, y_train_ave, y_test_ave, y_train_std, y_test_std\
            = Generate_train_test_data(df, train_idx, test_idx, embedding)
        cv_scores = {}
        
        ori_b_list = b_list

        for kernel_name in kernel_list:
            
            if kernel_name == 'Spectrum_Kernel' or 'WD_Kernel':
                b_list = [1]
            else:
                b_list = ori_b_list
                
            for alpha in alpha_list:
                for l_list in l_lists:
                    for b in b_list:
                        scores = []
                        kernel = kernel_dict[kernel_name]
                        b = float(b)

                        if kernel_name == 'Spectrum_Kernel' or 'WD_Kernel': 
                            gp_reg = GaussianProcessRegressor(kernel = kernel(l_list = l_list, weight_flag = weight_flag, padding_flag = padding_flag, gap_flag = gap_flag), alpha = alpha)
                        elif kernel_name == 'Sum_Spectrum_Kernel':
                            gp_reg = GaussianProcessRegressor(kernel = kernel(l_list = l_list, b = b, weight_flag = weight_flag, padding_flag = padding_flag, gap_flag = gap_flag), alpha = alpha)
                        else: 
                            gp_reg = GaussianProcessRegressor(kernel = kernel(), alpha = alpha)

                        for train_train_idx, train_val_idx in Train_val_split(len(train_idx), cv= cv, random_state=random_state):
                            train_train_df, train_val_df, X_train_train, X_train_val, y_train_train_sample, y_train_val_sample, y_train_train_ave, y_train_val_ave, y_train_train_std, y_train_val_std \
                                = Generate_train_test_data(df, train_train_idx, train_val_idx, embedding)
                            gp_reg.fit(X_train_train, y_train_train_sample)
                            y_train_val_predict = gp_reg.predict(X_train_val)
                            if eva_on_ave_flag:
                                scores.append(eva_metric(y_train_val_ave, y_train_val_predict)) # evaluate on AVERAGE value
                            else:
                                scores.append(eva_metric(y_train_val_sample, y_train_val_predict)) # evaluate on samples
                        #scores = cross_val_score(gp_reg, X_train, y_train, cv = cv, scoring = make_scorer(eva_metric))
            
                        cv_scores[kernel_name + '-' + str(alpha)+ '-' + json.dumps(l_list) + '-' + str(b)] = np.asarray(scores).mean() 
        
        fig, ax = plt.subplots()
        fig.set_size_inches(12,8)
        sns.scatterplot(list(cv_scores.keys()), list(cv_scores.values()), ax = ax, marker = '.')
        ax.set_xticklabels(list(cv_scores.keys()), rotation = 90)
        plt.xlabel('kernel, alpha')
        plt.ylabel(str(eva_metric))
        plt.show()

        if eva_metric == r2_score:
            optimal_kernel, optimal_alpha, optimal_l_list, optimal_b = list(cv_scores.keys())[np.argmax(list(cv_scores.values()))].split('-')
        else:
            optimal_kernel, optimal_alpha, optimal_l_list, optimal_b = list(cv_scores.keys())[np.argmin(list(cv_scores.values()))].split('-')
        optimal_l_list = json.loads(optimal_l_list)
        print('optimal kernel: ', optimal_kernel, ', optimal alpha: ', optimal_alpha, ', optiaml l list: ', optimal_l_list, ', optimal b: ', optimal_b)
        
        optimal_kernel = str(optimal_kernel)
        print(optimal_kernel)
        if optimal_kernel == 'Spectrum_Kernel' or 'WD_Kernel': 
            print('1')
            gp_reg = GaussianProcessRegressor(kernel = kernel_dict[optimal_kernel](l_list = optimal_l_list, weight_flag = weight_flag, padding_flag = padding_flag, gap_flag = gap_flag), alpha = float(optimal_alpha))
        elif optimal_kernel == 'Sum_Spectrum_Kernel':
            print('2')
            gp_reg = GaussianProcessRegressor(kernel = kernel_dict[optimal_kernel](l_list = optimal_l_list, b = float(optimal_b), weight_flag = weight_flag, padding_flag = padding_flag, gap_flag = gap_flag), alpha = float(optimal_alpha))
        else:
            print('3')
            gp_reg = GaussianProcessRegressor(kernel = kernel_dict[optimal_kernel](), alpha = float(optimal_alpha))
            
        gp_reg.fit(X_train,y_train_sample)
        y_train_pred = gp_reg.predict(X_train)
        y_test_pred= gp_reg.predict(X_test)

        if eva_on_ave_flag:
            print('Train: ', eva_metric(y_train_ave, y_train_pred))
            print('Test: ', eva_metric(y_test_ave, y_test_pred))
            test_scores.append(eva_metric(y_test_ave, y_test_pred))

            plt.scatter(y_train_ave, y_train_pred, label = 'train')
            plt.scatter(y_test_ave, y_test_pred, label = 'test')
            
        else:
            print('Train: ', eva_metric(y_train_sample, y_train_pred))
            print('Test: ', eva_metric(y_test_sample, y_test_pred))
            test_scores.append(eva_metric(y_test_sample, y_test_pred))

            plt.scatter(y_train_sample, y_train_pred, label = 'train')
            plt.scatter(y_test_sample, y_test_pred, label = 'test')
        plt.xlabel('label')
        plt.ylabel('pred')
        plt.legend()
        plt.plot([-2, 3], [-2,3])
        plt.show()
    print('Cross-validation Test mean: ', np.asarray(test_scores).mean())
    print('Cross-validation Test std: ', np.asarray(test_scores).std())
        
    return optimal_alpha, test_scores
    



    