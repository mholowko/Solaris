import numpy as np
from collections import defaultdict, OrderedDict
import operator
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error

from codes.environment import Rewards_env
from codes.kernels import spectrum_kernel, mixed_spectrum_kernel, WD_kernel, WD_shift_kernel

KERNEL_TYPE = {'spectrum': spectrum_kernel,
               'mixed_spectrum': mixed_spectrum_kernel,
               'WD': WD_kernel,
               'WD_shift': WD_shift_kernel
            }

class Regression():
    """Regression.

    Attributes
    --------------------------------------------------------
    model: instance of regression class (from sklearn)
            attribute: kernel (if 'precomputed', use precomputed kernel matrix)
    X: array
        features array (num_samples, ) 
        first column of data, each element is a string
    Y: array
        labels array (num_samples, )
        second column of data, each element is a float/int
    """
    def __init__(self, model, data, embedding_method = None, 
                 cp = False, split_idx = None, cross_val_flag = True):
        """
        Paramters:
        ------------------------------------------------------
        model: instance of regression class (from sklearn)
            attribute: kernel (if 'precomputed', use precomputed kernel matrix)
        data: ndarray 
            num_data * 2
            two columns: biology sequence; score (label)
        embedding_method: string, default is None
            if None, no embedding is performed and set X to the first column of data
        cp: boolean, default is False
            true: cross prediction, split data to data_train and data_test according to split_idx
                take subset of data_train as training set
            false: train_test_split or cross validation
        split_idx: int, defalut is None
            set only is cp = True
            data_train = data[: split_idx]
            data_test = data[split_idx: ]
        cross_val_flag: boolean, 
            True: cross validation
            False: single run prediction
        """
        self.model = model
        self.cross_val_flag = cross_val_flag
        self.cp = cp
        self.split_idx = split_idx
        self.num_seq = data.shape[0]
        
        if embedding_method is not None:
            self.my_env = Rewards_env(data, embedding_method)
            self.X = self.my_env.embedded
        else:
            self.X = data[:, 0]
        self.Y = data[:, 1]

        self.test_scores1 = []
        self.test_scores2 = [] # only use for cross prediction
        

    def run_k(self, k = 10, train_per = 0.9):
        if self.cp:
            if self.split_idx == None:
                print('Please specify split idx. Split half-half.')
                self.split_idx = self.X.shape[0]/2
            X_train = self.X[: self.split_idx]
            self.X_test = self.X[self.split_idx :]
            Y_train = self.Y[: self.split_idx]
            self.Y_test = self.Y[self.split_idx :]

            train_size = X_train.shape[0]
            subset_size = int(train_size * train_per)

            for i in range(k):
                subset_idxs = np.random.choice(train_size, subset_size, replace=False)
                self.X_train = X_train[subset_idxs]
                self.Y_train = Y_train[subset_idxs]
                self.test_scores1.append(self.run_single())

            X_train = self.X[self.split_idx :] 
            self.X_test = self.X[: self.split_idx]
            Y_train = self.Y[self.split_idx :] 
            self.Y_test = self.Y[: self.split_idx]

            train_size = X_train.shape[0]
            subset_size = int(train_size * train_per)

            for i in range(k):
                subset_idxs = np.random.choice(train_size, subset_size, replace=False)
                self.X_train = X_train[subset_idxs]
                self.Y_train = Y_train[subset_idxs]
                self.test_scores2.append(self.run_single())

        else:
            kf = KFold(n_splits= k, shuffle=True, random_state=42)
            for (train_idx, test_idx) in kf.split(self.X):
                self.X_train, self.X_test = self.X[train_idx], self.X[test_idx]
                self.Y_train, self.Y_test = self.Y[train_idx], self.Y[test_idx]
                self.test_scores1.append(self.run_single())
 
    def run_single(self):
        """Evaluate.
        Calculate RMSE score for both training and testing datasets.
        """
        self.model.fit(self.X_train, self.Y_train)
        train_predict = self.model.predict(self.X_train)
        test_predict = self.model.predict(self.X_test)

        #train_rmse = np.sqrt(mean_squared_error(self.Y_train, train_predict))
        test_rmse = np.sqrt(mean_squared_error(self.Y_test, test_predict))
        return test_rmse

'''

    def plot(self):    
        """Plot for predict vs. true label. 
        """   
        plt.figure() 
        plt.plot(self.test_predict, self.Y_test, 'r.', label = 'test')
        plt.plot(self.train_predict, self.Y_train, 'b.', label = 'train')
        max_label = max(self.Y)
        min_label = min(self.Y)
        plt.plot([min_label,max_label], [min_label,max_label], '--')
        plt.plot([min_label,max_label], [max_label/2.0,max_label/2.0], 'k--')
        plt.plot([max_label/2.0,max_label/2.0], [min_label,max_label], 'k--')
        
        # plt.plot([0,1], [0,1], '--')
        # plt.plot([0,1], [0.5,0.5], 'k--')
        # plt.plot([0.5,0.5], [0,1], 'k--')

        plt.xlabel('Prediction')
        plt.ylabel('True Label')
        plt.title(str(self.model))
        plt.xlim(min_label,max_label)
        plt.ylim(min_label,max_label)
        plt.legend()
    
'''
        

   

    