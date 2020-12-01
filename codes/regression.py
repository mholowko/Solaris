# update: 25/Nov/2020
# use unique seq for kernel normalisation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import plotly
import plotly.graph_objs as go
import seaborn as sns
import itertools
from collections import defaultdict
import math
import json
import xarray as xr  

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel, DotProduct, RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans, AgglomerativeClustering

from codes.embedding import Embedding
from codes.environment import Rewards_env
from codes.ucb import GPUCB, Random
from codes.evaluations import evaluate, plot_eva
from codes.kernels_for_GPK import *
from codes.sort_seq import sort_kernel_matrix
from codes.plot_format import *

from ipywidgets import IntProgress
import codes.config # provide configuration, e.g. global variable, path

# Aug 2020 Mengyan Zhang
# Implement predictors based on Gaussian Process Regression

KERNEL_DICT = {
    # 'Spectrum_Kernel': Spectrum_Kernel,
    # 'Mixed_Spectrum_Kernel': Mixed_Spectrum_Kernel,
    # 'WD_Kernel': WeightedDegree_Kernel,
    # 'Sum_Spectrum_Kernel': Sum_Spectrum_Kernel,
    'WD_Kernel_Shift': WD_Shift_Kernel,
    'RBF': RBF  
}

REP_LIST = ['Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5','Rep6', 'Rep7', 'Rep8','Rep9']


class GPR_Predictor():
    def __init__(self, df, test_size=0.2, train_idx = None, test_idx = None, 
                 kernel_name='WD_Kernel_Shift', l=6, s = 1, b=0.33, sigma_0 = 1, padding_flag=False, gap_flag=False,
                 alpha=2, embedding='label', eva_metric=[r2_score, mean_squared_error],  eva_on = "samples",
                 kernel_norm_flag = True, centering_flag = True, unit_norm_flag = True,
                #  kernel_norm_flag = True,　centering_flag = True,　unit_norm_flag = True,
                 ):
        """
        Parameter
        --------------------------------------------------------
        df: dataframe. 
            Input dataframe of Seq dataset: [Name, Group, RBS, RBS6, Rep1...RepN,AVERAGE,STD]
        test_size: float
            indicate test size, used when train_idx and test_idx are not specified.
        train_idx: list of idx
            indicating training data
        test_idx: list of idx 
            indicating testing data

        * Paramters for kernel

        kernel_name: string
            indicates kernel to use
        l: int
            length of lmer, i.e. substring length
        s: int
            shift length, only used for WD_Kernel_Shift
        b: float
            weight for sum of spectrum kernel
        sigma_0: float
            signal std; value multiply to normalised kernel,
            i.e. https://drafts.distill.pub/gp/#section-4.2
        padding_flag: Boolean, default False
            indicates whether adding padding characters before and after sequences
        gap_flag: Boolean, default False
            indicates whether generates substrings with gap

        alpha: float
            GPR parameter, values added to the diagonal
        embedding: embedding method
            to generate features
        # TODO: change the code for these two design
        eva_metric: list of possible eva_metric
            default is r2_score and mean_square_error
        eva_on: string
            indicates evaluating on samples or seqs

        kernel_norm_flag: boolean
            indicates whether to use kernel normalisation
        centering_flag: boolean
            indicates whether to do kernel centering
        unit_norm_flag: boolean
            indicates whether to do kernel unit norm normalisation
        """
        self.df = df
        self.test_size = test_size
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.num_data = self.df.shape[0]

        self.kernel_name = kernel_name
        self.kernel = KERNEL_DICT[kernel_name]
        self.kernel_norm_flag = kernel_norm_flag
        self.centering_flag = centering_flag
        self.unit_norm_flag = unit_norm_flag
        # self.kernel.INIT_FLAG = False 
        # so that kernel will be initialised (cal_kernel)
        self.l = l
        self.s = s
        self.b = b
        self.sigma_0 = sigma_0
        self.padding_flag = padding_flag
        self.gap_flag = gap_flag
        
        self.alpha = alpha
        self.embedding = embedding
        # if self.embedding == 'label':
        #     self.encoder = Embedding().label()
        self.eva_metric = eva_metric
        self.eva_on = eva_on

    def Train_test_split(self, random_state = 24):
        np.random.seed(random_state)
        self.test_idx = np.random.choice(self.num_data, int(self.test_size * self.num_data), replace=False)
        self.train_idx = list(set(range(self.num_data)) - set(self.test_idx))

        self.train_idx = np.asarray(self.train_idx)
        self.test_idx = np.asarray(self.test_idx)

    def Train_val_split(self, cv = 5, random_state = 24):
        kf = KFold(n_splits = cv, shuffle = True, random_state = random_state)
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
        use_samples_for_train = True
        if use_samples_for_train:
            # self.df['index'] = self.df.index
            train_df = pd.melt(self.df.loc[self.train_idx], id_vars=['idx','RBS', 'RBS6', 'AVERAGE', 'STD', 'Group'], 
                               value_vars=REP_LIST)
            train_df = train_df.dropna(subset=['RBS', 'AVERAGE', 'value'])
            self.train_df = train_df.rename(columns = {'value': 'label'})
        else:
            self.train_df = self.df.loc[self.train_idx].copy()
            self.train_df['label'] = self.train_df['AVERAGE'].copy()

        if self.eva_on == 'samples':
        # if True:
            test_df = pd.melt(self.df.loc[self.test_idx], id_vars=['idx','RBS', 'RBS6', 'AVERAGE', 'STD', 'Group'], 
                               value_vars=REP_LIST)
            test_df = test_df.dropna()
            self.test_df = test_df.rename(columns = {'value': 'label'})

            y_test_sample = np.asarray(self.test_df['label'])
        else: # seq
            self.test_df = self.df.loc[self.test_idx].copy()
            self.test_df['label'] = self.test_df['AVERAGE'].copy()

            y_test_sample = None

        # TODO: make the embedding code consistent for label and onehot
        # if self.embedding == 'label':
        #     X_train = np.asarray[self.encoder.transform(list(self.train_df['RBS'])[i])\
        #                             for i in range(len(self.train_df))]) 
        # else:   
        X_train = Rewards_env(np.asarray(self.train_df[['RBS', 'label']]), self.embedding).embedded
        y_train_sample = np.asarray(self.train_df['label'])
        y_train_ave = np.asarray(self.train_df['AVERAGE'])
        if 'STD' in self.train_df.columns:
            y_train_std = np.asarray(self.train_df['STD'])
        else:
            y_train_std = None
        
        # if self.embedding == 'label':
        #     X_test = np.asarray[self.encoder.transform(list(self.test_df['RBS'])[i])\
        #                             for i in range(len(self.test_df))]) 
        # else: 
        X_test = Rewards_env(np.asarray(self.test_df[['RBS', 'AVERAGE']]), self.embedding).embedded
        y_test_ave = np.asarray(self.test_df['AVERAGE']) 
        if 'STD' in self.test_df.columns:
            y_test_std = np.asarray(self.test_df['STD'])
        else:
            y_test_std = None
        
        return X_train, X_test, y_train_sample, y_test_sample, y_train_ave, y_test_ave, y_train_std, y_test_std

    def regression(self, random_state = 24):
        """Regression with train and test splitting build-in, i.e. test size as 0.2.
        """
        
        if self.train_idx is None and self.test_idx is None :
            self.Train_test_split(random_state) # update train and test idx using random split
        else:
            self.test_size = len(self.test_idx)/(len(self.train_idx)+ len(self.test_idx))

        X_train, X_test, y_train_sample, y_test_sample, y_train_ave, y_test_ave, y_train_std, y_test_std=self.Generate_train_test_data()
        print('X train shape: ', X_train.shape)
        print('X test shape: ', X_test.shape)

        # self.kernel.INIT_FLAG = False # compute kernel
        # self.features = np.concatenate((X_train,  X_test), axis = 0)

        if self.kernel_name == 'WD_Kernel_Shift':
            print('create kernel instance')
            self.wd_kernel_instance = self.kernel(l = self.l, s = self.s, sigma_0=self.sigma_0, 
                                        kernel_norm_flag = self.kernel_norm_flag, centering_flag = self.centering_flag, unit_norm_flag = self.unit_norm_flag)
            kernel_instance = self.wd_kernel_instance \
                            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-5, 1e+5))
            # debug
            self.kernel_instance = kernel_instance
            print('finish creating kernel instance')
            self.gp_reg = GaussianProcessRegressor(kernel = kernel_instance, alpha = self.alpha, n_restarts_optimizer = 0)

        # TODO: implement other kernels
        # elif self.kernel_name == 'Sum_Spectrum_Kernel':
        #     self.gp_reg = GaussianProcessRegressor(kernel = self.kernel(l_list = self.l_list, features = self.features, test_size = self.test_size, b = self.b), alpha = self.alpha)
        # else:
        #     self.gp_reg = GaussianProcessRegressor(kernel = self.kernel(l_list = self.l_list, features = self.features, test_size = self.test_size,), alpha = self.alpha)
        elif self.kernel_name == 'RBF':
            self.gp_reg = GaussianProcessRegressor(kernel = self.kernel(length_scale = 1), alpha = self.alpha, n_restarts_optimizer = 5)
            
        print('gp_reg fit')
        self.gp_reg.fit(X_train,y_train_sample)
        print('gp_reg pred') 

        y_train_pred_mean, y_train_pred_std = self.gp_reg.predict(X_train, return_std=True)
        # print('regression train pred mean ', y_train_pred_mean)
        y_test_pred_mean, y_test_pred_std = self.gp_reg.predict(X_test, return_std=True)

        # try_hete_gp = True
        # if try_hete_gp:
        #     import GPy
        #     print(np.asarray(self.train_df['index']))
        #     self.gp_reg = GPy.models.GPHeteroscedasticRegression(X_train,y_train_sample.reshape(-1,1), Y_metadata={'output_index': np.asarray(self.train_df['index'])})
        #     y_train_pred_mean, y_train_pred_std = self.gp_reg.predict(X_train)
        #     # print('regression train pred mean ', y_train_pred_mean)
        #     y_test_pred_mean, y_test_pred_std = self.gp_reg.predict(X_test)
        #     y_train_pred_std = np.sqrt(y_train_pred_std)
        #     y_test_pred_std = np.sqrt(y_test_pred_std)

        self.train_df['pred mean'] = y_train_pred_mean
        self.test_df['pred mean'] = y_test_pred_mean
        self.train_df['pred std'] = y_train_pred_std
        self.test_df['pred std'] = y_test_pred_std
        print('finish reg')

    def scatter_plot(self, plot_format = 'plt', title = 'Prediction'):
        """Scatter plot for predictions.
        x-axis: label
        y-axis: prediction
        """
        if self.eva_on == 'samples':
            eva_column = 'label'
        else:
            eva_column = 'AVERAGE'

        # if eva_column == 'AVERAGE': # debug
        #     self.train_df = self.train_df[self.train_df['variable'] == 'Rep1']

        for metric in self.eva_metric:
            print(str(metric))
            print('Train: ', metric(self.train_df[eva_column], self.train_df['pred mean']))
            print('Test: ', metric(self.test_df[eva_column], self.test_df['pred mean']))

        # report slope
        test_pred_fit = np.polyfit(x = range(len(self.test_df)), y=self.test_df.sort_values(by = ['AVERAGE'])['pred mean'],deg=1)
        test_ave_fit = np.polyfit(x = range(len(self.test_df)), y=self.test_df.sort_values(by = ['AVERAGE'])['AVERAGE'],deg=1)
        print('Test pred fit: ', test_pred_fit)
        print('Test ave fit: ', test_ave_fit)
        
        if 'pred std' in self.test_df:
            print('coverage rate: ')
            print('Train: ',  self.coverage_rate(self.train_df[eva_column], self.train_df['pred mean'], self.train_df['pred std']))
            print('Test: ',  self.coverage_rate(self.test_df[eva_column], self.test_df['pred mean'], self.test_df['pred std']))

        if plot_format == 'plt':
            plt.scatter(self.train_df[eva_column], self.train_df['pred mean'], label = 'train')
            plt.scatter(self.test_df[eva_column], self.test_df['pred mean'], label = 'test')
            plt.xlabel('label')
            plt.ylabel('pred')
            plt.legend()
            plt.plot([-2, 3], [-2,3])
            plt.title(title)
            plt.show()
        elif plot_format == 'plotly':
            train_scatter = go.Scatter(x = self.train_df[eva_column], y = self.train_df['prediction'], mode = 'markers', 
                        text = np.asarray(self.train_df['RBS']), name = 'train', hoverinfo='text')
            test_scatter = go.Scatter(x = self.test_df[eva_column], y = self.test_df['prediction'], mode = 'markers', 
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

        eva_column = 'label' 

        y_train_ave = np.asarray(self.train_df[eva_column])
        y_train_std = np.asarray(self.train_df['STD'])
        y_train_pred_mean = np.asarray(self.train_df['pred mean'])
        y_train_pred_std = np.asarray(self.train_df['pred std'])

        y_test_ave = np.asarray(self.test_df[eva_column])
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

    def sort_kernel_label_plot(self, df, title = 'Train Data Similarity and Prediction', sort_method = 'label_distance', group_flag = True):
        """Generate plot with kernel matrix and prediction sharing x-axis.
        Sequences are sorted in terms of label/similarity.
        """
        # new_kernel = KERNEL_DICT[self.kernel_name]
        # new_kernel.INIT_FLAG = False
        # feature_kernel = new_kernel(l=self.l, features = np.asarray(df['RBS'])).kernel_all_normalised
        feature_kernel = self.kernel_instance(np.asarray(df['RBS']))

        if not group_flag:
            df['Group'] = 'all' 
        
        # sort 
        feature_kernel, order = sort_kernel_matrix(df, feature_kernel, sort_method)

        f, ax = plt.subplots(figsize=(10,12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
        ax = plt.subplot(gs[0])
        im = ax.imshow(feature_kernel, origin='lower', aspect='auto')
        # f.colorbar(im, ax = ax)
        # plt.xlim(tlim)

        df = df.loc[order,:].reset_index()
        # axl = plt.subplot(gs[0,0], sharey=ax)
        axb = plt.subplot(gs[1], sharex=ax)
        axb.plot(df.index, df['pred mean'], label = 'Pred Mean')
        axb.fill_between(df.index, df['pred mean'] + 1.96 * df['pred std'],df['pred mean'] - 1.96 * df['pred std'], alpha = 0.5)
        axb.scatter(df[df['train_test'] == 'train'].index, df[df['train_test'] == 'train']['AVERAGE'], s =1, c='green', label = 'Train Label')
        axb.scatter(df[df['train_test'] == 'test'].index, df[df['train_test'] == 'test']['AVERAGE'], s =1, c='red', label = 'Test Label')
        # axb.plot(t, a.mean(0))
        axb.set_xlabel('RBS Sequences')
        axb.set_ylabel('TIR Normalised Label')
        axb.legend()
        ax.set_title(valid_name(title))
        # plt.show()
        plt.savefig(valid_path(title) + '.pdf', bbox_inches='tight')
        plt.close()

    def coverage_rate(self, true_label, pred_mean, pred_std):
        """evaluation metric of prediction
        the percent of data points inside of 95% predicted confidence interval (pred mean +/- 1.96 std)
        """
        coverage_count = 0
        # print('true label: ', true_label)
        # print('pred mean: ', pred_mean)
        # print('pred std: ', pred_std)
        true_label = np.asarray(true_label)
        pred_mean = np.asarray(pred_mean)
        pred_std = np.asarray(pred_std)

        for i in range(len(true_label)):
            ucb = pred_mean[i] + 1.96 * pred_std[i]
            lcb = pred_mean[i] - 1.96 * pred_std[i]
            if true_label[i]<=ucb and true_label[i] >= lcb:
                coverage_count+=1
        # print('coverage count: ', coverage_count)
        # print('coverage rate: ', float(coverage_count)/float(len(true_label)))
        return float(coverage_count)/float(len(true_label))


    # cross validation on training dataset. Find the optimal alpha. Double loop.

    def Repeated_kfold(self, num_split = 5, num_repeat = 10, 
                       kernel_norm_flag = [True],
                       centering_flag = [True],
                       unit_norm_flag = [True],
                       alpha_list = [2], rbf_lengthscale_list = [1],
                       l_list = [3], s_list = [0], 
                       sigma_0_list = [1],
                       eva_on_list = ['samples', 'seq'],
                       eva_metric_list = [mean_squared_error, r2_score, 'coverage rate']):
        """Repeated kfold for hyparameter choosing.

        Parameters
        --------------------------------------------------------
        num_split: int
            k of kfold, k-1 fold for training, the left one fold for testing
        num_repeat: int
            repeat with different random state for splitting
        
        alpha_list: list
            parameter for GPR, value added to the diagonal of kernel
        l_list: list
            lmer, number of substring
        s_list: list
            number of shift
        sigma_0_list
            hyperparameter for kernel, signal std
        eva_on: list
            evaluating on samples or seq (average)
        eva_metric: list
            mean square error of r2 score


        Return
        ---------------------------------------------------------
        use xarray to store results
        dimensions:
        
        train_test: results for train or test 
        eva_on: evaluate on sample labels or average
        eva_metric: mean square error or r2 score
        alpha (parameter of GPR, which adds to the diagonal of kernel matrix)
        l (length of kmer)
        s (shift length)
        sigma_0 (signal std, kernel hyper)
        repeat (nth repeat)
        fold (k-fold)
        """
        print('Repeated KFold Running ...')
        max_count = num_repeat*num_split*len(kernel_norm_flag)*len(centering_flag)*len(unit_norm_flag)*len(alpha_list) * len(rbf_lengthscale_list) * len(l_list)*len(s_list) * len(sigma_0_list)
        f = IntProgress(min=0, max=max_count) # instantiate the bar
        display(f) # display the bar
    
        # init of xarray elements
        result_data = np.zeros((2, len(eva_on_list), len(eva_metric_list), len(kernel_norm_flag), len(centering_flag), len(unit_norm_flag),\
                            len(alpha_list), len(rbf_lengthscale_list), len(l_list), len(s_list), len(sigma_0_list), num_repeat, num_split))

        train_scores = defaultdict(list)
        test_scores = defaultdict(list)

        for repeat_idx in range(num_repeat):
            for kernel_norm_idx, kernel_norm in enumerate(kernel_norm_flag):
                for centering_idx, centering in enumerate(centering_flag):
                    for unit_norm_idx, unit_norm in enumerate(unit_norm_flag):
                        for alpha_idx, alpha in enumerate(alpha_list):
                            for lengthscale_idx, lengthscale in enumerate(rbf_lengthscale_list):
                                for l_idx, l in enumerate(l_list):
                                    for s_idx, s in enumerate(s_list):
                                        for sigma_0_idx, sigma_0 in enumerate(sigma_0_list):
                                            cv = 0
                                            for train_idx, test_idx in self.Train_val_split(cv = num_split, random_state=repeat_idx):
                                                self.train_idx = train_idx
                                                self.test_idx = test_idx
                                                X_train, X_test, y_train_sample, y_test_sample, y_train_ave, y_test_ave, y_train_std, y_test_std = self.Generate_train_test_data()
                                                
                                                # self.kernel.INIT_FLAG = False
                                                # self.features = np.concatenate((X_train,  X_test), axis = 0)
                                                if self.kernel_name == 'WD_Kernel_Shift':
                                                    kernel_instance = self.kernel(l = l, 
                                                                                # features = self.features, 
                                                                                # n_train=X_train.shape[0], n_test=X_test.shape[0],
                                                                                s = s, sigma_0= sigma_0, kernel_norm_flag = kernel_norm,
                                                                                centering_flag = centering, unit_norm_flag = unit_norm) \
                                                                    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-5, 1e+5))
                                                    gp_reg = GaussianProcessRegressor(kernel = kernel_instance, alpha = alpha, n_restarts_optimizer = 0)
                                                elif self.kernel_name == 'RBF':
                                                    gp_reg = GaussianProcessRegressor(kernel = self.kernel(length_scale = lengthscale), alpha = self.alpha, n_restarts_optimizer = 3)
                                                # else:
                                                #     # TODO: tidy up other cases
                                                #     gp_reg = GaussianProcessRegressor(kernel = self.kernel(l_list = l_list, features = self.features, test_size = self.test_size), alpha = alpha)
                                            
                                                gp_reg.fit(X_train, y_train_sample) # train with samples
                                                y_train_predict, y_train_predict_uncertainty = gp_reg.predict(X_train, return_std=True)
                                                y_test_predict, y_test_predict_uncertainty = gp_reg.predict(X_test, return_std=True)
                                                
                                                for i, eva_on in enumerate(eva_on_list):
                                                    for j, eva_metric in enumerate(eva_metric_list):
                                                        if eva_on == 'samples':
                                                            if eva_metric == 'coverage rate':
                                                                result_data[0, i, j, kernel_norm_idx, centering_idx, unit_norm_idx, alpha_idx, lengthscale_idx, l_idx, s_idx, sigma_0_idx, repeat_idx, cv] =\
                                                                    self.coverage_rate(y_train_sample, y_train_predict, y_train_predict_uncertainty)
                                                                result_data[1, i, j,  kernel_norm_idx, centering_idx, unit_norm_idx, alpha_idx, lengthscale_idx, l_idx, s_idx, sigma_0_idx, repeat_idx, cv] =\
                                                                    self.coverage_rate(y_train_sample, y_train_predict, y_train_predict_uncertainty)
                                                            else:
                                                                result_data[0, i, j,  kernel_norm_idx, centering_idx, unit_norm_idx, alpha_idx, lengthscale_idx, l_idx, s_idx, sigma_0_idx, repeat_idx, cv] = eva_metric(y_train_sample, y_train_predict)
                                                                result_data[1, i, j,  kernel_norm_idx, centering_idx, unit_norm_idx, alpha_idx, lengthscale_idx, l_idx, s_idx, sigma_0_idx, repeat_idx, cv] = eva_metric(y_test_sample, y_test_predict)
                                                        else:
                                                            if eva_metric == 'coverage rate':
                                                                result_data[0, i, j,  kernel_norm_idx, centering_idx, unit_norm_idx, alpha_idx, lengthscale_idx, l_idx, s_idx, sigma_0_idx, repeat_idx, cv] =\
                                                                    self.coverage_rate(y_train_ave, y_train_predict, y_train_predict_uncertainty)
                                                                result_data[1, i, j,  kernel_norm_idx, centering_idx, unit_norm_idx, alpha_idx, lengthscale_idx, l_idx, s_idx, sigma_0_idx, repeat_idx, cv] =\
                                                                    self.coverage_rate(y_train_ave, y_train_predict, y_train_predict_uncertainty)
                                                            else:
                                                                result_data[0, i, j,  kernel_norm_idx, centering_idx, unit_norm_idx, alpha_idx, lengthscale_idx, l_idx, s_idx, sigma_0_idx, repeat_idx, cv] = eva_metric(y_train_ave, y_train_predict)
                                                                result_data[1, i, j,  kernel_norm_idx, centering_idx, unit_norm_idx, alpha_idx, lengthscale_idx, l_idx, s_idx, sigma_0_idx, repeat_idx, cv] = eva_metric(y_test_ave, y_test_predict)
                                                
                                                cv += 1
                                                f.value+=1 # visualise progress
                                                print(f.value)

                        #train_scores[kernel_name + '-' + str(alpha)+ '-' + json.dumps(l_list) + '-' + str(b)].append(np.asarray(train_fold_scores).mean())
                        #test_scores[kernel_name + '-' + str(alpha)+ '-' + json.dumps(l_list) + '-' + str(b)].append(np.asarray(test_fold_scores).mean() )

        # l_lists_coord = []
        # for l_list in l_lists:
        #     l_lists_coord.append(str(l_list))

        result_DataArray = xr.DataArray(
                                result_data, 
                                coords=[['Train', 'Test'], eva_on_list, eva_metric_list,  kernel_norm_flag, centering_flag, unit_norm_flag, alpha_list, rbf_lengthscale_list, l_list, s_list, sigma_0_list, range(num_repeat), range(num_split)], 
                                dims=['train_test', 'eva_on', 'eva_metric', 'kernel_norm_flag', 'centering_flag', 'unit_norm_flag', 'alpha', 'rbf_lengthscale', 'l', 's', 'sigma_0', 'num_repeat', 'num_split']
                                )
        
        # result_DataArray.attrs['eva_metric'] = self.eva_metric
        
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


class KRR_Predictor(GPR_Predictor):
    from sklearn.kernel_ridge import KernelRidge
    # TODO: Repeated_KFold needs to be modified as well

    def regression(self, random_state = 24):
        """Regression with train and test splitting build-in, i.e. test size as 0.2.
        """

        if self.train_idx is None and self.test_idx is None :
            self.Train_test_split(random_state) # update train and test idx using random split
        else:
            self.test_size = len(self.test_idx)/(len(self.train_idx)+ len(self.test_idx))

        X_train, X_test, y_train_sample, y_test_sample, y_train_ave, y_test_ave, y_train_std, y_test_std=self.Generate_train_test_data()
        print('X train shape: ', X_train.shape)
        print('X test shape: ', X_test.shape)

        self.kernel.INIT_FLAG = False # compute kernel
        self.features = np.concatenate((X_train,  X_test), axis = 0)

        if self.kernel_name == 'WD_Kernel_Shift':
            print('create kernel instance')
            self.wd_kernel_instance = self.kernel(l = self.l, 
                                            s = self.s, sigma_0=self.sigma_0, kernel_norm_flag = self.kernel_norm_flag,
                                            centering_flag = self.centering_flag, unit_norm_flag = self.unit_norm_flag)
            kernel_instance = self.wd_kernel_instance \
                            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-5, 1e+5))
            # debug
            self.kernel_instance = kernel_instance
            print('finish creating kernel instance')
            self.gp_reg = KernelRidge(kernel = kernel_instance, alpha = self.alpha)

        # TODO: implement other kernels
        # elif self.kernel_name == 'Sum_Spectrum_Kernel':
        #     self.gp_reg = GaussianProcessRegressor(kernel = self.kernel(l_list = self.l_list, features = self.features, test_size = self.test_size, b = self.b), alpha = self.alpha)
        # else:
        #     self.gp_reg = GaussianProcessRegressor(kernel = self.kernel(l_list = self.l_list, features = self.features, test_size = self.test_size,), alpha = self.alpha)
        elif self.kernel_name == 'RBF':
            self.gp_reg = KernelRidge(kernel = self.kernel(length_scale = 1), alpha = self.alpha)
            
        print('gp_reg fit')
        self.gp_reg.fit(X_train,y_train_sample)
        print('gp_reg pred') 

        y_train_pred_mean = self.gp_reg.predict(X_train)
        # print('regression train pred mean ', y_train_pred_mean)
        y_test_pred_mean = self.gp_reg.predict(X_test)

        # try_hete_gp = True
        # if try_hete_gp:
        #     import GPy
        #     print(np.asarray(self.train_df['index']))
        #     self.gp_reg = GPy.models.GPHeteroscedasticRegression(X_train,y_train_sample.reshape(-1,1), Y_metadata={'output_index': np.asarray(self.train_df['index'])})
        #     y_train_pred_mean, y_train_pred_std = self.gp_reg.predict(X_train)
        #     # print('regression train pred mean ', y_train_pred_mean)
        #     y_test_pred_mean, y_test_pred_std = self.gp_reg.predict(X_test)
        #     y_train_pred_std = np.sqrt(y_train_pred_std)
        #     y_test_pred_std = np.sqrt(y_test_pred_std)

        self.train_df['pred mean'] = y_train_pred_mean
        self.test_df['pred mean'] = y_test_pred_mean
        # self.train_df['pred std'] = y_train_pred_std
        # self.test_df['pred std'] = y_test_pred_std
        print('finish reg')

