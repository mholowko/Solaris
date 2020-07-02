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
from codes.kernels_for_GPK import Spectrum_Kernel, Sum_Spectrum_Kernel

# rewrite the regression code, including
# 1. regression with passed training/testing data, return predicted mean and std,
#       where the returned predicted mean and std can be used as the input of UCB algorithms
# 2. regression with the whole dataset, with splitting data method of function.
# 3. cross validation regression, to select hyperparameters

# For all classes,
# take the predictor, embedding, kernel as input

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
    def __init__(self, model, kernel, data, data_split=None, embedding_method='label', **kwargs):
        """
        Parameters:
        ------------------------------------------------------
        model: instance of regression class (from sklearn)
            e.g. GaussianProcessRegressor
        kernel: instance from kernels_for_GPK.py
            e.g. SpecturmKernel
        data: ndarray 
            num_data, 2
            two columns: biology sequence; score (label)
        data_split: list 
            list of idx for training and testing data,
            e.g. [[0,1,2],[3,4]], means 0,1,2 for training and 3,4 for testing.
        embedding_method: string, default is None
            if None, no embedding is performed and set X to the first column of data
        """
        
        self.kernel = kernel
        self.env = Rewards_env(data, embedding_method=embedding_method)
        self.X = self.env.embedded
        self.Y = data[:,1]
        self.data_split = data_split

        # hyperparameters
        self.l_list = kwargs.get('l_list', [2,3,4,5,6]) # default kmer (k=3)
        self.alpha = kwargs.get('alpha', 0.05)

        self.model = model(kernel = self.kernel(l_list = self.l_list), alpha = self.alpha)
        # TODO: add others

    def split_data(self):
        """Split data into training and testing datasets. 
        """
        if self.data_split == None:
            self.X_train, self.X_test, self.Y_train, self.Y_test = \
                train_test_split(self.X, self.Y, test_size = 0.2, random_state = 24)
        else:
            self.X_train = self.X[np.asarray(self.data_split[0])]
            self.X_test = self.X[np.asarray(self.data_split[1])]

            self.Y_train = self.Y[np.asarray(self.data_split[0])]
            self.Y_test = self.Y[np.asarray(self.data_split[1])]

    def train(self):
        """Train.
        """
        self.split_data()
        self.model.fit(self.X_train, self.Y_train)
        self.train_predict = self.model.predict(self.X_train)
        self.test_predict = self.model.predict(self.X_test)

    def evaluate(self, print_flag = True, plot_flag = True, metric = 'RMSE'):
        """Evaluate.
        Calculate RMSE score for both training and testing datasets.
        """

        if metric is 'RMSE':
            # use the normalised root mean square error
            train_score = np.sqrt(mean_squared_error(self.Y_train, self.train_predict))
            test_score = np.sqrt(mean_squared_error(self.Y_test, self.test_predict))
        elif metric is 'r2_score':
            train_score = r2_score(self.Y_train, self.train_predict)
            test_score =  r2_score(self.Y_test, self.test_predict)

        if print_flag:
            print('Model: ', str(self.model))
            print('Train '+metric+ ': '+ str(train_score))
            print('Test '+metric+ ': '+ str(test_score))
        if plot_flag:
            self.plot()
        
        return train_score, test_score

    def plot(self):    
        """Plot for predict vs. true label. 
        """   
        plt.figure() 
        plt.plot(self.test_predict, self.Y_test, 'r.', label = 'test')
        plt.plot(self.train_predict, self.Y_train, 'b.', label = 'train')
        max_label = max(list(self.Y_train) + list(self.Y_test))
        min_label = min(list(self.Y_train) + list(self.Y_test))
        plt.plot([min_label - 0.5,max_label+ 0.5], [min_label - 0.5,max_label+ 0.5], '--')
        plt.plot([min_label - 0.5,max_label+ 0.5], [(max_label + min_label)/2.0,(max_label + min_label)/2.0], 'k--')
        plt.plot([(max_label + min_label)/2.0,(max_label + min_label)/2.0], [min_label - 0.5,max_label+ 0.5], 'k--')
        
        # plt.plot([0,1], [0,1], '--')
        # plt.plot([0,1], [0.5,0.5], 'k--')
        # plt.plot([0.5,0.5], [0,1], 'k--')

        plt.xlabel('Prediction')
        plt.ylabel('True Label')
        plt.title(str(self.model))
        plt.xlim(min_label,max_label)
        plt.ylim(min_label,max_label)
        plt.legend()
    

        

   

    