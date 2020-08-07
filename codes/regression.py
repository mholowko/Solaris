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

KERNEL_DICT = {
    'Spectrum_Kernel': Spectrum_Kernel,
    'Mixed_Spectrum_Kernel': Mixed_Spectrum_Kernel,
    'WD_Kernel': WeightedDegree_Kernel,
    'Sum_Spectrum_Kernel': Sum_Spectrum_Kernel,
    'WD_Kernel_Shift': WD_Shift_Kernel
    
}

class GPR_Predictor():
    def __init__(self, df, test_size=0.2, train_idx = None, test_idx = None, 
                 kernel_name='WD_Kernel', normalise_kernel = False, alpha=0.5, embedding='label',
                 eva_metric=r2_score, l_list=[3], s = 0, b=0.33, 
                 weight_flag=False, padding_flag=False, gap_flag=False):
        """
        Parameter
        --------------------------------------------------------
        df: dataframe. 
            Input dataframe of Seq dataset: [Name, Group, RBS, RBS6, Rep1...RepN,AVERAGE,STD]
        train_idx: list of idx
            indicating training data
        test_idx: list of idx 
            indicating testing data
        embedding: embedding method
            to generate features
        normalise_kernel: boolean
            #TODO: only used for WD shift kernel, only for "regression"
            True indicates normalise kernel over the whole kernel to get unit norm
            Phi is normalised no matter what True or False
        """
        self.df = df
        self.test_size = test_size
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.kernel_name = kernel_name
        self.kernel = KERNEL_DICT[kernel_name]
        self.normalise_kernel = normalise_kernel
        self.alpha = alpha
        self.embedding = embedding
        self.eva_metric = eva_metric
        self.l_list = l_list
        self.s = s
        self.b = b
        self.weight_flag = weight_flag
        self.padding_flag = padding_flag
        self.gap_flag = gap_flag

        self.num_data = self.df.shape[0]


    def Train_test_split(self, random_state = 24):
        np.random.seed(random_state)
        self.test_idx = np.random.choice(self.num_data, int(self.test_size * self.num_data), replace=False)
        self.train_idx = list(set(range(self.num_data)) - set(self.test_idx))

        self.train_idx = np.asarray(self.train_idx)
        self.test_idx = np.asarray(self.test_idx)

    def Train_val_split(self, cv = 5, random_state = 24):
        kf = KFold(n_splits = cv, shuffle = True)
        return kf.split(range(self.num_data))

    def Generate_train_test_data(self):
        """Generate train test data for group data.
        The seperation should guarantee:
            - splitting according to unique sequences. 
            Sequences in testing data should not show in the training data.
            - training with individual samples
            - testing on unique sequences
            - evaluation on averaged label

        Return
        -------------------------------------------------------------
        train_df: melt train df
        test_df: df with test idx
        X_train: feature of training samples
        X_test: features of testing sequences
        y_train_sample: training sample labels
        y_train_ave: training sample averaged labels
        y_test_ave: averaged labels of testing sequences
        y_train_std: std of training samples
        y_test_std: std of testing sequences
        """
        
        train_df = pd.melt(self.df.loc[self.train_idx], id_vars=['RBS', 'RBS6', 'AVERAGE', 'STD', 'Group'], value_vars=['Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5'])
        train_df = train_df.dropna(subset=['RBS', 'AVERAGE', 'value'])
        self.train_df = train_df.rename(columns = {'value': 'label'})

        #test_df = pd.melt(df.loc[test_idx], id_vars=['RBS', 'RBS6', 'AVERAGE', 'STD', 'Group'], value_vars=['Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5'])
        #test_df = test_df.dropna()
        #test_df = test_df.rename(columns = {'value': 'label'})
        
        self.test_df = self.df.loc[self.test_idx]
            
        X_train = Rewards_env(np.asarray(self.train_df[['RBS', 'label']]), self.embedding).embedded
        y_train_sample = np.asarray(self.train_df['label'])
        y_train_ave = np.asarray(self.train_df['AVERAGE'])
        if 'STD' in self.train_df.columns:
            y_train_std = np.asarray(self.train_df['STD'])
        else:
            y_train_std = None
        
        X_test = Rewards_env(np.asarray(self.test_df[['RBS', 'AVERAGE']]), self.embedding).embedded
        #y_test_sample = np.asarray(test_df['label'])
        y_test_ave = np.asarray(self.test_df['AVERAGE']) 
        if 'STD' in self.test_df.columns:
            y_test_std = np.asarray(self.test_df['STD'])
        else:
            y_test_std = None
        
        return X_train, X_test, y_train_sample, y_train_ave, y_test_ave, y_train_std, y_test_std

    def regression(self, random_state = 24):
        """Regression with train and test splitting build-in, i.e. test size as 0.2.
        """
        
        if self.train_idx is None and self.test_idx is None :
            self.Train_test_split(random_state) # update train and test idx using random split

        X_train, X_test, y_train_sample, y_train_ave, y_test_ave, y_train_std, y_test_std=self.Generate_train_test_data()
        print('X train shape: ', X_train.shape)
        print('X test shape: ', X_test.shape)
        if self.kernel_name == 'WD_Kernel_Shift':
            self.gp_reg = GaussianProcessRegressor(kernel = self.kernel(l_list = self.l_list, s = self.s, normalise_flag = self.normalise_kernel), alpha = self.alpha)
        elif self.kernel_name == 'Sum_Spectrum_Kernel':
            self.gp_reg = GaussianProcessRegressor(kernel = self.kernel(l_list = self.l_list, b = self.b), alpha = self.alpha)
        else:
            self.gp_reg = GaussianProcessRegressor(kernel = self.kernel(l_list = self.l_list), alpha = self.alpha)

        self.gp_reg.fit(X_train,y_train_sample)
        y_train_pred_mean, y_train_pred_std = self.gp_reg.predict(X_train, return_std=True)
        y_test_pred_mean, y_test_pred_std = self.gp_reg.predict(X_test, return_std=True)

        self.train_df['pred mean'] = y_train_pred_mean
        self.test_df['pred mean'] = y_test_pred_mean
        self.train_df['pred std'] = y_train_pred_std
        self.test_df['pred std'] = y_test_pred_std

    def scatter_plot(self, plot_format = 'plt'):
        """Scatter plot for predictions.
        x-axis: label
        y-axis: prediction
        """
        
        print('Train: ', self.eva_metric(self.train_df['AVERAGE'], self.train_df['pred mean']))
        print('Test: ', self.eva_metric(self.test_df['AVERAGE'], self.test_df['pred mean']))

        if plot_format == 'plt':
            plt.scatter(self.train_df['AVERAGE'], self.train_df['pred mean'], label = 'train')
            plt.scatter(self.test_df['AVERAGE'], self.test_df['pred mean'], label = 'test')
            plt.xlabel('label')
            plt.ylabel('pred')
            plt.legend()
            plt.plot([-2, 3], [-2,3])
            plt.show()
        elif plot_format == 'plotly':
            train_scatter = go.Scatter(x = self.train_df['AVERAGE'], y = self.train_df['prediction'], mode = 'markers', 
                        text = np.asarray(self.train_df['RBS']), name = 'train', hoverinfo='text')
            test_scatter = go.Scatter(x = self.test_df['AVERAGE'], y = self.test_df['prediction'], mode = 'markers', 
                        text = np.asarray(self.test_df['RBS']), name = 'test', hoverinfo='text')
            diag_plot = go.Scatter(x = [-2, 3.5], y = [-2,3.5], name = 'diag')
            layout = go.Layout(xaxis_title = 'label', yaxis_title= 'pred')
            fig = go.Figure(data=[train_scatter, test_scatter, diag_plot], layout=layout)
            
            fig.show()
        
    def line_plot(self):
        """Line plot for regression.
        Sort according to the labels. 
        Show one std for both the labels (std in terms of several samples/replicates) 
        and predictions (in terms of the posterior uncertainty) 
        """
        y_train_ave = np.asarray(self.train_df['AVERAGE'])
        y_train_std = np.asarray(self.train_df['STD'])
        y_train_pred_mean = np.asarray(self.train_df['pred mean'])
        y_train_pred_std = np.asarray(self.train_df['pred std'])

        y_test_ave = np.asarray(self.test_df['AVERAGE'])
        y_test_std = np.asarray(self.test_df['STD'])
        y_test_pred_mean = np.asarray(self.test_df['pred mean'])
        y_test_pred_std = np.asarray(self.test_df['pred std'])

        argsort_train_ave_idx = np.asarray(np.argsort(y_train_ave))
        
        plt.plot(range(len(y_train_ave)), np.asarray(y_train_pred_mean)[argsort_train_ave_idx], label = 'y_train_pre_mean')
        plt.plot(range(len(y_train_ave)), np.asarray(y_train_ave)[argsort_train_ave_idx], label = 'y_train_ave')
        plt.fill_between(range(len(y_train_ave)), 
                        np.asarray(y_train_pred_mean)[argsort_train_ave_idx] + np.asarray(y_train_pred_std)[argsort_train_ave_idx],
                        np.asarray(y_train_pred_mean)[argsort_train_ave_idx] - np.asarray(y_train_pred_std)[argsort_train_ave_idx],
                        label = 'y_train_pred_std', alpha = 0.2)
        plt.fill_between(range(len(y_train_ave)), 
                        np.asarray(y_train_ave)[argsort_train_ave_idx] + np.asarray(y_train_std)[argsort_train_ave_idx],
                        np.asarray(y_train_ave)[argsort_train_ave_idx] - np.asarray(y_train_std)[argsort_train_ave_idx],
                        label = 'y_train_std', alpha = 0.2)
        plt.legend()
        plt.show()
        
        argsort_test_ave_idx = np.asarray(np.argsort(y_test_ave))
        
        plt.plot(range(len(y_test_ave)), np.asarray(y_test_pred_mean)[argsort_test_ave_idx], label = 'y_test_pre_mean')
        plt.plot(range(len(y_test_ave)), np.asarray(y_test_ave)[argsort_test_ave_idx], label = 'y_test_ave')
        plt.fill_between(range(len(y_test_ave)), 
                        np.asarray(y_test_pred_mean)[argsort_test_ave_idx] + np.asarray(y_test_pred_std)[argsort_test_ave_idx],
                        np.asarray(y_test_pred_mean)[argsort_test_ave_idx] - np.asarray(y_test_pred_std)[argsort_test_ave_idx],
                        label = 'y_test_pred_std', alpha = 0.2)
        plt.fill_between(range(len(y_test_ave)), 
                        np.asarray(y_test_ave)[argsort_test_ave_idx] + np.asarray(y_test_std)[argsort_test_ave_idx],
                        np.asarray(y_test_ave)[argsort_test_ave_idx] - np.asarray(y_test_std)[argsort_test_ave_idx],
                        label = 'y_test_pred_std', alpha = 0.2)
        plt.legend()
        plt.show()

    # cross validation on training dataset. Find the optimal alpha. Double loop.

    def Repeated_kfold(self, num_split = 5, num_repeat = 10, alpha_list = [0.1, 1], l_lists = [[3]], s_list = [0]):
        """Regression with repeated kfold.

        use xarray to store results
        dimensions:
        
        train_test: results for train or test 
        alpha (parameter of GPR, which adds to the diagonal of kernel matrix)
        l (length of kmer)
        s (shift length)
        repeat (nth repeat)
        fold (k-fold)
        """
        
        random_state_list = list(range(num_repeat))
    
        # init of xarray elements
        result_data = np.zeros((2, len(alpha_list), len(l_lists), len(s_list), num_repeat, num_split))

        train_scores = defaultdict(list)
        test_scores = defaultdict(list)

        for repeat_idx, random_state in enumerate(random_state_list):
            for alpha_idx, alpha in enumerate(alpha_list):
                for l_idx, l_list in enumerate(l_lists):
                    for s_idx, s in enumerate(s_list):
                        
                        if self.kernel_name == 'WD_Kernel_Shift':
                            gp_reg = GaussianProcessRegressor(kernel = self.kernel(l_list = l_list, s = s), alpha = alpha)
                        else:
                            gp_reg = GaussianProcessRegressor(kernel = self.kernel(l_list = l_list), alpha = alpha)
                        
                        cv = 0
            
                        for train_idx, test_idx in self.Train_val_split(cv = num_split, random_state = random_state):
                            self.train_idx = train_idx
                            self.test_idx = test_idx
                            X_train, X_test, y_train_sample, y_train_ave, y_test_ave, y_train_std, y_test_std = self.Generate_train_test_data()
                            
                            gp_reg.fit(X_train, y_train_sample) # train with samples
                            y_train_predict = gp_reg.predict(X_train)
                            y_test_predict = gp_reg.predict(X_test)
                            
                            result_data[0, alpha_idx, l_idx, s_idx, repeat_idx, cv] = self.eva_metric(y_train_ave, y_train_predict)
                            result_data[1, alpha_idx, l_idx, s_idx, repeat_idx, cv] = self.eva_metric(y_test_ave, y_test_predict)
                            #train_fold_scores.append(eva_metric(y_train_ave, y_train_predict)) # evaluate on AVERAGE value
                            #test_fold_scores.append(eva_metric(y_test_ave, y_test_predict)) # evaluate on AVERAGE value
            
                            
                            cv += 1
                        #train_scores[kernel_name + '-' + str(alpha)+ '-' + json.dumps(l_list) + '-' + str(b)].append(np.asarray(train_fold_scores).mean())
                        #test_scores[kernel_name + '-' + str(alpha)+ '-' + json.dumps(l_list) + '-' + str(b)].append(np.asarray(test_fold_scores).mean() )

        l_lists_coord = []
        for l_list in l_lists:
            l_lists_coord.append(str(l_list))

        result_DataArray = xr.DataArray(
                                result_data, 
                                coords=[['Train', 'Test'], alpha_list, l_lists_coord, s_list, range(num_repeat), range(num_split)], 
                                dims=['train_test', 'alpha', 'l', 's', 'num_repeat', 'num_split']
                                )
        
        result_DataArray.attrs['eva_metric'] = self.eva_metric
        
        '''
        fig, ax = plt.subplots()
        fig.set_size_inches(12,8)
        sns.scatterplot(list(train_scores.keys()), list(np.mean(train_scores.values())), ax = ax, marker = '.', color = blue)
        sns.scatterplot(list(train_scores.keys()), list(np.mean(train_scores.values()) + np.std(train_scores.values())), ax = ax, marker = '*', color = orange)
        sns.scatterplot(list(train_scores.keys()), list(np.mean(train_scores.values()) - np.std(train_scores.values())), ax = ax, marker = '*', color = orange)
        ax.set_xticklabels(list(test_scores.keys()), rotation = 90)
        plt.xlabel('kernel, alpha')
        plt.ylabel(str(eva_metric))
        plt.title('Performance on Training data (Repeated KFold)')
        plt.show()

        fig, ax = plt.subplots()
        fig.set_size_inches(12,8)
        sns.scatterplot(list(test_scores.keys()), list(np.mean(test_scores.values())), ax = ax, marker = '.', color = blue)
        sns.scatterplot(list(test_scores.keys()), list(np.mean(test_scores.values()) + np.std(test_scores.values())), ax = ax, marker = '*', color = orange)
        sns.scatterplot(list(test_scores.keys()), list(np.mean(test_scores.values()) - np.std(test_scores.values())), ax = ax, marker = '*', color = orange)
        ax.set_xticklabels(list(test_scores.keys()), rotation = 90)
        plt.xlabel('kernel, alpha')
        plt.ylabel(str(eva_metric))
        plt.title('Performance on Testing data (Repeated KFold)')
        plt.show()
        '''
        return result_DataArray

    def Regression_for_ucb(self):
        """Regression for ucb. 
        Training set as all available data.
        
        """