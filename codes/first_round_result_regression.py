#!/usr/bin/env python
# coding: utf-8

# In[1]:
    
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
#import seaborn as sns
from itertools import product
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel, DotProduct, RBF 
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error

from codes.embedding import Embedding
from codes.environment import Rewards_env
from codes.ucb import GPUCB, Random
from codes.evaluations import evaluate, plot_eva
from codes.regression import Regression
from codes.kernels import spectrum_kernel
from codes.kernels_pairwise import spectrum_kernel_pw, sum_spectrum_kernel_pw, mixed_spectrum_kernel_pw, WD_kernel_pw, WD_shift_kernel_pw

from ipywidgets import IntProgress
from IPython.display import display
import warnings



# ## Reading Dataset

# In[2]:


# Data contains both the first round result and baseline data
#
# columns: 
# RBS: 20-length RBS seq
# RBS6: 6-length RBS seq, which the [7:13] positions
# variable: for first round results, denote which replication is the seq. 
# label: normalised label (zero mean and unit variance)
#       For the first round result, label is the (GFPOD(t0 + h) - GFPOR(t0))/h,
#                                   where t0 is the turning time point
#                                         h is the time interval (e.g. 4h)
#       For the baseline data, label is the TIR used in previous paper
#       Both of the two labels express the slope, but in different scales (calculation methods are slightly different)
#       We normalise the labels as zero mean and unit variance respectively

Path = 'data/firstRound_4h+Baseline.csv'

df = pd.read_csv(Path)
df.head()


# In[3]:


df.shape


# In[4]:


plt.hist(df['label'])
plt.title('All label')


# In[5]:


# extract first round result
df_FRR = df[df['Group'] == 'First round result']
plt.hist(df_FRR['label'])
plt.title('Fisrt round label')
df_FRR.shape


# In[6]:


# extract baseline data
df_BD = df[df['Group'] == 'Baseline data']
plt.hist(df_BD['label'])
plt.title('Baseline label')
df_BD.shape


# In[7]:


# data6: num_data * 2, columns: [6-base RBS seq (D), TIR labels (C)]
data = np.asarray(df[['RBS', 'label']])
data_FRR = np.asarray(df_FRR[['RBS', 'label']])
data_BD = np.asarray(df_BD[['RBS', 'label']])


# In[8]:


# indicates whether cross validation (KFOLD)
cross_val_flag = False 

# indicates whether plot predict_label vs. true label
plot_flag = True 

# regression evaluation metric, 'NRMSE' or 'r2_score'
metric = 'NRMSE'

# string kernel list 
# kernels = [spectrum_kernel_pw, mixed_spectrum_kernel_pw, WD_kernel_pw, WD_shift_kernel_pw]
kernels = [sum_spectrum_kernel_pw]

models = [GaussianProcessRegressor]

# name dictionaries
regression_name = {KernelRidge: 'KR',
                  GaussianProcessRegressor: 'GPR'}
embedding_name = ['onehot', 'kmer', 'label']
kernel_name = {
               spectrum_kernel_pw: 'spec',
               sum_spectrum_kernel_pw: 'sspec',
               #sum_onehot_spectrum_kernel_pw: 'sospec',
               #mixed_spectrum_kernel_pw: 'mspec',
               #WD_kernel_pw: 'WD',
               #sum_onehot_WD_kernel_pw: 'sWD',
               #WD_shift_kernel_pw: 'WDshift'
                }

train_scores = {}
test_scores = {}
trained_reg_model_dict = {}

data_dict = {
             'all': data,
             'First round result': data_FRR,
             'Baseline data': data_BD,
             None: None
            }

alpha = 1e-10 # Value added to the diagonal of the kernel matrix during fitting. 


# In[9]:


def run_regression(model, data_name, kernel, embedding, alpha, data_name_test = None):
    # when data_name_test is none, the regression is run on the data_name with splitting to 80/20
    # otherwise, the regression model is trained on data_name and tested on data_name_test
    if model == KernelRidge:
        reg = Regression(model(kernel = kernel, alpha = alpha), data_dict[data_name], data_dict[data_name_test], embedding)
    elif model == GaussianProcessRegressor:
        reg = Regression(model(kernel = PairwiseKernel(metric = kernel), alpha = alpha, normalize_y = True), data_dict[data_name], data_dict[data_name_test], embedding)
    
    key = data_name + '_' + str(data_name_test) + '_' + regression_name[model] + '_' + kernel_name[kernel] + '_' + embedding
    trained_model = reg.train()
    trained_reg_model_dict[key] = trained_model
    
    train_score, test_score = reg.evaluate(cross_val_flag = cross_val_flag, plot_flag = plot_flag, metric = metric)
    
    
    train_scores[key] = train_score
    test_scores[key] = test_score
    print()


# ## Train_Rep12_Test_Rep3

# The idea of this notebook is to train with Rep 1 and 2, then test with Rep 3. The label of Rep3 should inside of the confidence interval of prediction. Assuming the distribution of the replication is Gaussian,
# $$p=F(\mu+n \sigma)-F(\mu-n \sigma)$$
# when n = 1, p = 0.68; n = 2, p = 0.95. But since we do not have enough replications for a sequence, we evaluate the percentage using all sequences. That is, when we set n = 2, about 95% of Rep3 label of all sequences should be inside of prediction interval (well, it is not equivalent, but it's approximately correct).

# In[10]:


#Path = '../../data/firstRound_4h.csv'

#df_frr = pd.read_csv(Path).dropna()
alpha_list = [0.1]


# In[11]:


df.head()


# In[11]:


def rep_cross_predict(df, alpha_list, train_on = ['Rep1', 'Rep2'], test_on = ['Rep3'], n = 2):
    # df_Rep_train = pd.melt(df_frr[['RBS'] + train_on], id_vars = ['RBS'], value_vars = train_on)
    # data_Rep_train = np.asarray(df_Rep_train[['RBS', 'value']])

    # data_Rep_test = np.asarray(df_frr[['RBS'] + test_on])
    if len(train_on) == 1:
        data_Rep_train = np.asarray(df[df['variable'] == train_on[0]][['RBS', 'label']])
    else:
        data_Rep_train = np.asarray(df[df['variable'] == train_on[0]].append(df[df['variable'] == train_on[1]])[['RBS', 'label']])
    data_Rep_test = np.asarray(df[df['variable'] == test_on[0]][['RBS', 'label']])
    
    # plot with sorted idx average true value of first round sequences
    sorted_idx = np.argsort(df[df['variable'] == test_on[0]]['AVERAGE'])

    for model in [GaussianProcessRegressor]:
        for kernel in kernels:
            for alpha in alpha_list:
                #run_regression(model, 'Rep12', kernel, 'label', 'Rep3')

                reg = Regression(model(kernel = PairwiseKernel(metric = kernel), alpha = alpha, normalize_y = False), data_Rep_train, data_Rep_test, 'label')
                print('train:')
                my_model = reg.train()
                print('predict:')
                mu, std = my_model.predict(reg.X_test, return_std = True)
                print('mu:', mu)
                print('std:', std)
                
                plt.figure()
                plt.plot(range(len(mu)), mu[sorted_idx], label='prediction', color = 'green', alpha = 0.5)
                plt.fill_between(range(len(mu)), (mu +  n * std)[sorted_idx], (mu - n * std)[sorted_idx], label = 'confidence width', color = 'orange', alpha = 0.5)
                plt.scatter(range(len(mu)), data_Rep_test[:,1][sorted_idx], label = str(test_on), color = 'blue', s =2)
                plt.title('alpha: '+str(alpha))
                plt.legend()
                plt.show()

                num_good_pred = 0
                for i,label in enumerate(data_Rep_test[:,1]):
                    if label >= mu[i] - n * std[i] and label <= mu[i] + n * std[i]:
                        num_good_pred += 1

                print('the rate of test rep labels inside of confidence width: ',float(num_good_pred)/data_Rep_test.shape[0])

                NRMSE = np.sqrt(mean_squared_error(data_Rep_test[:,1], mu))/(max(data_Rep_test[:,1]) - min(data_Rep_test[:,1]))
                print('NRMSE: ', NRMSE)


# In[12]:


rep_cross_predict(df, [0.1], train_on = ['Rep1', 'Rep2'], test_on = ['Rep3'], n = 2)