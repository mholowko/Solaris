import numpy as np
from collections import defaultdict, OrderedDict
import operator
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error

from codes.environment import Rewards_env
from codes.kernels import spectrum_kernel, mixed_spectrum_kernel, WD_kernel, WD_shift_kernel

KERNEL_TYPE = {'spectrum': spectrum_kernel,
               'mixed_spectrum': mixed_spectrum_kernel,
               'WD': WD_kernel,
               'WD_shift': WD_shift_kernel
            }

class Regression_CP():
    """Regression for cross prediction.
       Use one dataset to train the predictor and another one to evaluate.

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
    def __init__(self, model, data_train, data_test, embedding_method = None, 
                 precomputed_kernel = None):
        """
        Paramters:
        ------------------------------------------------------
        model: instance of regression class (from sklearn)
            attribute: kernel (if 'precomputed', use precomputed kernel matrix)
        data_train: ndarray     
            num_data * 2
            two columns: biology sequence; score (label)
        data_test: ndarray     
            num_data * 2
            two columns: biology sequence; score (label)
        embedding_method: string, default is None
            if None, no embedding is performed and set X to the first column of data
        precomputed_kernel: string, default is None
            must be the key of KERNEL_TYPE dict
        """
        self.model = model
        
        if embedding_method is not None:
            self.my_env = Rewards_env(data_train, embedding_method)
            self.X_train = self.my_env.embedded
        else:
            self.X_train = data_train[:, 0]
        self.Y_train = data_train[:, 1]

        if embedding_method is not None:
            self.my_env = Rewards_env(data_test, embedding_method)
            self.X_test = self.my_env.embedded
        else:
            self.X_test = data_test[:, 0]
        self.Y_test = data_test[:, 1]

    def train(self):
        """Train.
        """
        self.model.fit(self.X_train, self.Y_train)
        self.train_predict = self.model.predict(self.X_train)
        self.test_predict = self.model.predict(self.X_test)

    def evaluate(self, cross_val_flag = True, print_flag = True, plot_flag = True, k = 10, metric = 'neg_mean_squared_error'):
        """Evaluate.
        Calculate RMSE score for both training and testing datasets.
        """
        if cross_val_flag:
            cv = ShuffleSplit(n_splits=k, test_size=0.2, random_state=42)
            scores = cross_val_score(self.model, self.X, self.Y, cv = cv, scoring= metric)
            scores = np.sqrt(-scores)
 
        train_score = np.sqrt(mean_squared_error(self.Y_train, self.train_predict))
        test_score = np.sqrt(mean_squared_error(self.Y_test, self.test_predict))

        if print_flag:
            print('Model: ', str(self.model))
            if cross_val_flag:
                print(scores)
                print("RMSE : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            else:
                print('Train RMSE: ', train_score)
                print('Test RMSE: ', test_score)
        if plot_flag:
            self.plot()
        
        return train_score, test_score

    def plot(self):    
        """Plot for predict vs. true label. 
        """   
        plt.figure() 
        plt.plot(self.test_predict, self.Y_test, 'r.', label = 'test')
        plt.plot(self.train_predict, self.Y_train, 'b.', label = 'train')
        max_label = max(self.Y_train)
        min_label = min(self.Y_train)
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
    

        

   

    