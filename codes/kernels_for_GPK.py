import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn import preprocessing
#from strkernel.mismatch_kernel import MismatchKernel, preprocess
import matplotlib.pyplot as plt

import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# To be able to normalise the kernel matrix
# We cannot use Pairwise kernel for GP model
# So we will inherit from sklearn gaussain process kernel class
# https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/gaussian_process/kernels.py#L1213

# Normalise over all matrix 
# The split accordingly

from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

BASES = ['A','C','G','T']

# global features for all known and design data
# used for normalisation
Path = '../../data/known_design.csv'
df = pd.read_csv(Path)
FEATURES = np.asarray(df['RBS'])

class String_Kernel(Kernel):
    """Base class for string kernels
    
    Parameters
    ------------------
    l : list, default [3]
            list of number of l-mers (length of 'word')
            For example [2,3] means extracting all substrings of length 2 and 3 
    features: array
        list of features for all training and testing data
        used for normalisation over all data, refer to function [normalisation]
    weight_flag: Boolean, default False
        indicates whether manually build weights for phi
    padding_flag: Boolean, default False
        indicates whether adding padding characters before and after sequences
    gap_flag: Boolean, default False
        indicates whether generates substrings with gap
    # TODO: add ways to implement different gap methods

    # TODO: change hyperparameters maybe (now set to the same as Dotproduct)
    sigma_0 : float >= 0, default: 0.0
        Parameter controlling the inhomogenity of the kernel. If sigma_0=0,
        the kernel is homogenous.
    sigma_0_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on l
    """
    def __init__(self, l_list=[3], features = FEATURES, n_train = None, n_test = None,
                 weight_flag = False, padding_flag = False, gap_flag = False,
                 sigma_0=1e-10, sigma_0_bounds=(1e-10,1e10)):
        self.l_list = l_list
        self.weight_flag = weight_flag
        self.padding_flag = padding_flag
        self.gap_flag = gap_flag
        self.sigma_0 = sigma_0
        self.sigma_0_bounds = sigma_0_bounds

        # calculate the whole kernel matrix for the concatenate train and test data with normalisation
        # then slice the corresponding part for GPR

        # train n_train * d
        # test n_test * d
        # features (n_train + n_test) * d
        # kernel_all (n_train + n_test) * (n_train + n_test)

        self.n_train = n_train
        self.n_test = n_test
        self.features = features # train has to be put at the front

        self.kernel_all = self.cal_kernel(self.features, self.features)
        assert (self.kernel_all).all() == (self.kernel_all.T).all() # check symmetric

        # check positive definite
        # print('eignh for kernel_all:')
        # print(np.linalg.eigh(self.kernel_all))
        # L = np.linalg.cholesky(self.kernel_all)
        # print('kernel_all is positive definite')

        self.kernel_all_normalised = self.normalisation(self.kernel_all)
        
        # check positive definite
        # print('eignh for kernel_all_normalised:')
        # print(np.linalg.eigh(self.kernel_all_normalised))
        # L = np.linalg.cholesky(self.kernel_all_normalised)
        # print('kernel_all_normalised is positive definite')

    def cal_kernel(self, X, Y=None):
        """Calculate K(X,Y)
        """

    def __call__(self, X, Y=None):
        """Slice kernels for GPR
        """
    
    def normalisation(self, kernel):
        # https://jmlr.csail.mit.edu/papers/volume12/kloft11a/kloft11a.pdf
        # Sec 4.2.2
        # First calculate zero mean kernel
        # Then calculate unit variance
        # The the variance (defined as the 1/n trace - kernel mean) is 1 
        # normalise over the whole matrix (train, test)
        
        # centering
        # one_mat = np.matrix(np.ones(kernel.shape)) # kernel shape, i.e. (n_train+n_test, n_train+n_test)
        # one_vec = np.matrix(np.ones((kernel.shape[0],1))) # (n_train+n_test, 1)

        # row_sum = np.matrix(np.mean(kernel,axis=0)).T # (n_train+n_test, 1)
        # kernel_centered = kernel - row_sum * one_vec.T - one_vec * row_sum.T +\
        #     np.mean(row_sum.A)*one_mat

        # # unit variance
        
        # Kii = np.diag(kernel)
        # Kii.shape = (len(Kii),1)
        # kernel_centered_unit_norm =  np.divide(kernel, np.sqrt(np.matrix(Kii)*np.matrix(Kii).T))

        one_mat = np.ones(kernel.shape) # kernel shape, i.e. (n_train+n_test, n_train+n_test)
        one_vec = np.ones((kernel.shape[0],1)) # (n_train+n_test, 1)

        row_sum = np.mean(kernel,axis=0).T 
        row_sum = row_sum.reshape(row_sum.shape[0],1) # (n_train+n_test, 1)
        # print('kernel all')
        # print(kernel)
        # print('row sum:')
        # print(row_sum)
        kernel_centered = kernel - row_sum.dot(one_vec.T) - one_vec.dot(row_sum.T) +\
            np.mean(row_sum)*one_mat

        # kernel_centered += 1e-10 * np.identity(kernel_centered.shape[0])
        # print('eignh for kernel_centered:')
        # print(np.linalg.eigh(kernel_centered))
        # L = np.linalg.cholesky(kernel_centered)
        # print('kernel centered is positive definite')

        # unit variance
        
        Kii = np.diag(kernel_centered)
        Kii.shape = (len(Kii),1)
        kernel_centered_unit_norm =  np.divide(kernel_centered, np.sqrt(Kii*Kii.T))

        
        return kernel_centered_unit_norm   


    def Phi(self, X, Y, l_list = [3], j_X=0, j_Y=0, d=None, 
            weight_flag= False, padding_flag = False, gap_flag = False):
        """Calculate spectrum features for spectrum kernel.

        Phi is a mapping of the matrix X into a |alphabet|^l
        dimensional feature space. For each sequence in X,
        each dimension corresponds to one of the |alphabet|^l 
        possible strings s of length l and is the count of 
        the number of occurrance of s in x. 

        Paramters
        ---------------------------------------------------
        X : array of shape (n_samples_X, )
            each row is a sequence (string)
        Y : array of shape (n_samples_Y, )
            each row is a sequence (string)
        l : list, default [3]
            list of number of l-mers (length of 'word')
            For example [2,3] means extracting all substrings of length 2 and 3 
        j_X : int
            start position of sequence in X
        j_Y : int
            start position of sequence in Y
        d : int
            the length of analysed sequence
            j + d is end position of sequence 
        weight_flag: Boolean, default False
            indicates whether manually build weights for phi
        padding_flag: Boolean, default False
            indicates whether adding padding characters before and after sequences
        gap_flag: Boolean, default False
            indicates whether generates substrings with gap
        # TODO: implement different gap, padding, weight methods

        Returns
        ----------------------------------------------------
        embedded matrix of X: array
            num_X * num_embedded_features
        embedded matrix of Y
            num_Y * num_embedded_features
        """
        if d is None:
            d = len(X[0]) 

        num_X = X.shape[0]
        num_Y = Y.shape[0]

        sentences = []

        for i in range(num_X):
            words = []
            sequence= X[i][j_X:j_X + d]
            if padding_flag:
                sequence = 'ZZ' + sequence + 'ZZ' # Padding_flag
                #sequence = sequence[-2:] + sequence + sequence[:2] # Padding_flag
            for l in l_list:
                words += [sequence[a:a+l] for a in range(len(sequence) - l + 1)]
                if gap_flag:
                    words_gapped = generate_gapped_kmer(sequence, l)
                    words = words + words_gapped
            sentence = ' '.join(words)
            sentences.append(sentence)
        
        for i in range(num_Y):
            words = []
            sequence= Y[i][j_Y:j_Y + d]
            if padding_flag:
                sequence = 'ZZ' + sequence + 'ZZ' # Padding_flag
                #sequence = sequence[-2:] + sequence + sequence[:2] # Padding_flag
            for l in l_list:
                words += [sequence[a:a+l] for a in range(len(sequence) - l + 1)]
                if gap_flag:
                    words_gapped = generate_gapped_kmer(sequence, l)
                    words = words + words_gapped
            sentence = ' '.join(words)
            sentences.append(sentence)
        cv = CountVectorizer(analyzer='word',token_pattern=u"(?u)\\b\\w+\\b")
        #cv =  CountVectorizer()
        embedded = cv.fit_transform(sentences).toarray()
        embedded_X = embedded[: num_X, :].astype(float)
        embedded_Y = embedded[-num_Y: , :].astype(float)
        
        
        return embedded_X, embedded_Y
        
        #return embedded_X, embedded_Y

    def onehot(self, data):
        """One-hot embedding.

        data : array of shape (n_samples_X, )
            each row is a sequence (string)

        Returns
        --------------------------------------------
        embedded_data: ndarray
            {0, 1}^{num_seq x num_bases * 4}
        """
        
        base_dict = dict(zip(BASES,range(4))) # {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3}

        num_seq = data.shape[0]
        num_bases = len(data[0])
        embedded_data = np.zeros((num_seq, num_bases * 4))

        # loop through the array of sequences to create a feature array 
        for i in range(num_seq):
            seq = data[i]
            # loop through each individual sequence, from the 5' to 3' end
            for b in range(num_bases):
                embedded_data[i, b * 4 + base_dict[seq[b]]] = 1

        embedded_data -= np.nanmean(embedded_data, axis = 0)
        normalised_embedded_data = preprocessing.normalize(embedded_data, norm = 'l2')
        return normalised_embedded_data

    def generate_gapped_kmer(self, sequence, l):
        words_gapped = []
        for a in range(len(sequence) - l + 1):
            # Only works for l = 3 for now
            words_gapped.append(sequence[a] + 'N' + sequence[a+l-1]) 
        return words_gapped

    def inverse_label(self, X):
        """convert_to_string
        """
        le = preprocessing.LabelEncoder()
        le.fit(BASES)

        inversed_label_X = np.array(["".join(le.inverse_transform(list(X[i])))\
                                        for i in range(X.shape[0])])
        
        return inversed_label_X

    def weights_for_Phi(self, feature_names):
        # can be computational slow
        w_d = len(feature_names)
        W = np.zeros((w_d, w_d))
        for j in range(w_d): # phi y
            for i in range(w_d): # phi x
                weight = 0
                feature_i = feature_names[i]
                feature_j = feature_names[j]
                for idx in range(len(feature_i)):
                    # weights are designed for l=3, same strings have weight 1
                    if feature_i[idx] == feature_j[idx]:
                        weight += 1.0/3
                    elif (feature_i[idx] == 'A' and feature_j[idx] == 'T') or\
                        (feature_i[idx] == 'T' and feature_j[idx] == 'A') or\
                        (feature_i[idx] == 'C' and feature_j[idx] == 'G') or\
                        (feature_i[idx] == 'G' and feature_j[idx] == 'C'):
                        weight += 1.0/6
                W[i,j] = weight
        return W

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).
        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        #return np.einsum('ij,ij->i', X, X) + self.sigma_0 ** 2
        K = self.__call__(X)
        return K.diagonal().copy()

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False

    def __repr__(self):
        return "{0}(sigma_0={1:.3g})".format(
            self.__class__.__name__, self.sigma_0)

    def plot_kernel(self, K_dict, title = 'Spectrum Kernel Matrix'):
        """
        Parameters
        ----------------------
        K_dict:
            keys: name of kernel
            values: Kernel matrix
        """ 
        num_plot = len(K_dict)
        num_cols = min(2, num_plot)
        num_rows = int(np.ceil(float(num_plot)/num_cols))
        fig,a = plt.subplots(num_rows, num_cols)

        row_idx = 0
        col_idx = 0
        idx = 0
        for key, kernel in K_dict.items():

            if num_rows > 1 and num_cols > 1:
            
                im = a[row_idx][col_idx].imshow(kernel, cmap = 'viridis', interpolation='nearest')
                fig.colorbar(im, ax =a[row_idx][col_idx])
                a[row_idx][col_idx].set_title(key)
                
                if col_idx == num_cols -1:
                    row_idx += 1
                    col_idx = 0
                else:
                    col_idx += 1
            elif num_rows > 1 or num_cols > 1:
                im = a[idx].imshow(kernel, cmap = 'viridis', interpolation='nearest')
                fig.colorbar(im, ax =a[idx])
                a[idx].set_title(key)
                if idx < max(num_cols, num_rows) - 1:
                    idx += 1
            else:
                im = a.imshow(kernel, cmap = 'viridis', interpolation='nearest')
                fig.colorbar(im, ax =a)
                a.set_title(key)
                
        plt.title(title)
        plt.show()

                
class Spectrum_Kernel(Kernel):
    """
    
    Parameters
    ------------------
    l : list, default [3]
            list of number of l-mers (length of 'word')
            For example [2,3] means extracting all substrings of length 2 and 3 
    features: array
        list of features for all training and testing data
        used for normalisation over all data, refer to function [normalisation]
    weight_flag: Boolean, default False
        indicates whether manually build weights for phi
    padding_flag: Boolean, default False
        indicates whether adding padding characters before and after sequences
    gap_flag: Boolean, default False
        indicates whether generates substrings with gap
    # TODO: add ways to implement different gap methods

    # TODO: change hyperparameters maybe (now set to the same as Dotproduct)
    sigma_0 : float >= 0, default: 0.0
        Parameter controlling the inhomogenity of the kernel. If sigma_0=0,
        the kernel is homogenous.
    sigma_0_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on l
    """
    def __init__(self, l_list=[3], features = FEATURES, n_train = None, n_Test = None,
                 normalise_kernel_flag=True, weight_flag = False, padding_flag = False, gap_flag = False,
                 sigma_0=1e-10, sigma_0_bounds=(1e-10,1e10)):
        self.l_list = l_list
        self.weight_flag = weight_flag
        self.padding_flag = padding_flag
        self.gap_flag = gap_flag
        self.sigma_0 = sigma_0
        self.sigma_0_bounds = sigma_0_bounds

        # calculate the whole kernel matrix for the concatenate train and test data with normalisation
        # then slice the corresponding part for GPR

        # train n_train * d
        # test n_test * d
        # features (n_train + n_test) * d
        # kernel_all (n_train + n_test) * (n_train + n_test)

        self.n_train = n_train
        self.n_test = n_test
        self.features = features # train has to be put at the front

        self.kernel_all = self.__call__(self.features, self.features)
        assert (self.kernel_all).all() == (self.kernel_all.T).all() # check symmetric

        if normalise_kernel_flag:
            # print('calculating kernel_all')
            self.normalise_kernel_flag = False # only for calculate kernel_all
            
            
            self.kernel_all_mean0 = self.kernel_all.mean(axis = 0)
            self.kernel_all_mean1 = self.kernel_all.mean(axis = 1)
            assert (self.kernel_all_mean0).all() == (self.kernel_all_mean1).all()
            self.kernel_all_mean = self.kernel_all.mean()
        
            # print('kernel_all shape: ', self.kernel_all.shape)
        

        self.normalise_kernel_flag = normalise_kernel_flag

    @property
    def hyperparameter_sigma_0(self):
        return Hyperparameter("sigma_0", "numeric", self.sigma_0_bounds)
    
    def __call__(self, X, Y=None, j_X=0, j_Y=0, d=None, normalise_phi_flag = False, eval_gradient=False, print_flag = False, plot_flag = False):
        """
        Compute the spectrum kernel between X and Y:
            k_{l}^{spectrum}(x, y) = <phi(x), phi(y)>
        for each pair of rows x in X and y in Y.
        when Y is None, Y is set to be equal to X.

        Parameters
        ----------
        X : array of shape (n_samples_X, ) or (n_sample_X, n_num_features)
            each row is a sequence (string)
            Left argument of the returned kernel k(X, Y)

        Y : array of shape (n_samples_Y, ) or (n_sample_Y, n_num_features)
            each row is a sequence (string)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        kernel_matrix : array of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """

        if type(X[0,]) is not str and type(X[0,]) is not np.str_: 
            X = inverse_label(X)           

        if Y is None:
            Y = X
        elif type(Y[0,]) is not str  and type(Y[0,]) is not np.str_:
            Y = inverse_label(Y)

        if self.weight_flag:
            phi_X, phi_Y, W = Phi(X, Y, self.l_list, j_X, j_Y, d, normalise_phi_flag, weight_flag = self.weight_flag,
                                  padding_flag=self.padding_flag, gap_flag=self.gap_flag)
            K = np.normalisation(phi_X.dot(W).dot(phi_Y.T)) + self.sigma_0 ** 2
        else:
            phi_X, phi_Y = Phi(X, Y, self.l_list, j_X, j_Y, d, normalise_phi_flag, weight_flag = self.weight_flag, 
                                padding_flag=self.padding_flag, gap_flag=self.gap_flag)
            K = phi_X.dot(phi_Y.T) + self.sigma_0 ** 2

        if self.normalise_kernel_flag:
            K = self.normalisation(K)

        if plot_flag:
            self.plot_kernel({'K': K}, title = 'Spectrum Kernel Matrix')

        """
        print('Kernel matrix: ')
        print(K)

        # check unit variance and zero mean
        print('var of kernel matrix:')
        print(np.nanvar(K, axis = 0))
        print('mean of kernel matrix:')
        print(np.nanmean(K, axis = 0))
        """

        #assert (np.nanvar(K, axis = 0) == np.ones_like(K) + self.sigma_0 ** 2).all()
        #assert (np.nanmean(K, axis = 0) < np.ones_like(K) * 1e-10).all()

        if eval_gradient:
            if not self.hyperparameter_sigma_0.fixed:
                K_gradient = np.empty((K.shape[0], K.shape[1], 1))
                K_gradient[..., 0] = 2 * self.sigma_0 ** 2
                return K, K_gradient
            else:
                return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K

    def distance(self, X, Y=None, eval_gradient=False, print_flag = False, plot_flag = False):
        """
        
        Compute the distance between x and y:
        d(x,y) = sqrt(k(x,x) + k(y,y) - 2k(x,y))
            
        for each pair of rows x in X and y in Y.
        when Y is None, Y is set to be equal to X.

        Parameters
        ----------
        X : array of shape (n_samples_X, ) or (n_sample_X, n_num_features)
            each row is a sequence (string)
            Left argument of the returned kernel k(X, Y)

        Y : array of shape (n_samples_Y, ) or (n_sample_Y, n_num_features)
            each row is a sequence (string)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        kernel_matrix : array of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        K = self.__call__(X,Y)
        K_diag = K.diagonal()

        K1 = np.zeros((len(K_diag), len(K_diag)))
        K2 = np.zeros((len(K_diag), len(K_diag)))

        for i, diag in enumerate(K_diag):
            K1[i,:] = diag
            K2[:,i] = diag

        distance_matrix = np.sqrt(K1+K2-2*K)
        return  distance_matrix
        #return phi_X, phi_Y

    
    def normalisation(self, kernel):
        # https://jmlr.csail.mit.edu/papers/volume12/kloft11a/kloft11a.pdf
        # Sec 4.2.2
        # First calculate zero mean kernel
        # Then calculate unit variance
        # The the variance (defined as the 1/n trace - kernel mean) is 1 
        # normalise over the whole matrix (train, test)
        
        # TODO: fix the non positive definite caused by centering
        # zero mean
        standardized_kernel = np.zeros_like(kernel)

        if s0 == s1: # kernel over two same inputs      
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]): 
                    standardized_kernel[i,j] = kernel[i,j] - self.kernel_all_mean1[i] - self.kernel_all_mean0[j] + self.kernel_all_mean
        
        kernel = standardized_kernel
        assert (kernel).all() == (kernel.T).all()
        print('After centering')
        print(kernel.shape)
        #print(kernel)
        # if kernel.shape[0] == kernel.shape[1]:
        #     print('eigh:')
        #     print(np.linalg.eigh(kernel))
    
        # unit variance
        s0, s1 = kernel.shape
        spherical_kernel = np.zeros_like(kernel)
        if s0 == s1: # kernel over two same inputs
            # print('1')
            for i in range(s0):
                for j in range(s1):
                    spherical_kernel[i,j] = kernel[i,j]/np.sqrt(kernel[i,i] * kernel[j,j])
        elif (self.test_size < 0.5 and s0<s1) or (self.test_size >= 0.5 and s0>=s1): 
            # k(test, train) i -> i+s1 
            # print('2')
            for i in range(s0):
                for j in range(s1):
                    spherical_kernel[i,j] = kernel[i,j]/np.sqrt(self.kernel_all[i+s1, i+s1] * self.kernel_all[j,j])
        else: # k(train, test) j -> j+s0
            # print('3')
            for i in range(s0):
                for j in range(s1):
                    spherical_kernel[i,j] = kernel[i,j]/np.sqrt(self.kernel_all[i, i] * self.kernel_all[j+s0,j+s0])

        kernel = spherical_kernel
        if kernel.shape[0] == kernel.shape[1]: # TODO: only deal with the same inputs by now
            kernel += 1e-4 * np.identity(kernel.shape[0]) # avoid zero eigenvalue 
            print('eigh:')
            print(np.linalg.eigh(kernel))

        return kernel        

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).
        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        #return np.einsum('ij,ij->i', X, X) + self.sigma_0 ** 2
        K = self.__call__(X)
        return K.diagonal().copy()

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False

    def __repr__(self):
        return "{0}(sigma_0={1:.3g})".format(
            self.__class__.__name__, self.sigma_0)

    def plot_kernel(self, K_dict, title = 'Spectrum Kernel Matrix'):
        """
        Parameters
        ----------------------
        K_dict:
            keys: name of kernel
            values: Kernel matrix
        """ 
        num_plot = len(K_dict)
        num_cols = min(2, num_plot)
        num_rows = int(np.ceil(float(num_plot)/num_cols))
        fig,a = plt.subplots(num_rows, num_cols)

        row_idx = 0
        col_idx = 0
        idx = 0
        for key, kernel in K_dict.items():

            if num_rows > 1 and num_cols > 1:
            
                im = a[row_idx][col_idx].imshow(kernel, cmap = 'viridis', interpolation='nearest')
                fig.colorbar(im, ax =a[row_idx][col_idx])
                a[row_idx][col_idx].set_title(key)
                
                if col_idx == num_cols -1:
                    row_idx += 1
                    col_idx = 0
                else:
                    col_idx += 1
            elif num_rows > 1 or num_cols > 1:
                im = a[idx].imshow(kernel, cmap = 'viridis', interpolation='nearest')
                fig.colorbar(im, ax =a[idx])
                a[idx].set_title(key)
                if idx < max(num_cols, num_rows) - 1:
                    idx += 1
            else:
                im = a.imshow(kernel, cmap = 'viridis', interpolation='nearest')
                fig.colorbar(im, ax =a)
                a.set_title(key)
                
        plt.title(title)
        plt.show()

class WD_Shift_Kernel(String_Kernel):

    def __init__(self, l_list=[3], features = FEATURES, n_train = None, n_test = None,
                        weight_flag = False, padding_flag = False, gap_flag = False,
                        sigma_0=1e-10, sigma_0_bounds=(1e-10,1e10),
                        s = 0):
        """
        embedding_for_noncore: 'onehot' or 'phi'
            default is 'onehot'
        b: float
            the weight for K_B
        s: int
            shift length
        normalise_kernel_flag: boolean
            if True, normalise over the whole kernel 
        """
        self.s = s
        super().__init__(l_list, features, n_train, n_test,
                 weight_flag, padding_flag, gap_flag, sigma_0, sigma_0_bounds)

    def dotproduct_phi(self, X, Y, l_list, j_X, j_Y, d):
        phi_X, phi_Y = self.Phi(X, Y, l_list, j_X, j_Y, d, weight_flag = self.weight_flag, 
                                padding_flag=self.padding_flag, gap_flag=self.gap_flag)
        kernel = phi_X.dot(phi_Y.T) # + self.sigma_0 ** 2
        # kernel += 1e-20 * np.identity(kernel.shape[0])

        # check positive definite
        # L = np.linalg.cholesky(kernel)
        # print('kernel inside of wd is positive definite')
        # print('eignh for kernel inside of wd:')
        # print(np.linalg.eigh(kernel))
        return kernel

    def cal_kernel(self, X, Y=None, eval_gradient=False, print_flag = False, plot_flag = False):
        """Weighted degree kernel with shifts.
        Compute the mixed spectrum kernel between X and Y:
            K(x, y) = \sum_{d = 1}^{l} \sum_j^{L-d} \sum_{s=0 and s+j <= L}
                beta_d * gamma_j * delta_s *
                (k_d^{spectrum}(x[j+s:j+s+d],y[j:j+d]) + k_d^{spectrum}(x[j:j+d],y[j+s:j+s+d]))
        for each pair of rows x in X and y in Y.
        when Y is None, Y is set to be equal to X.

        beta_d = 2 frac{l - d + 1}{l^2 + 1}
        gamma_j = 1
        delta_s = 1/(2(s+1))

        TODO: to confirm why shift useful?
        
        Parameters
        ----------
        X : array of shape (n_samples_X, )
            each row is a sequence (string)
        Y : array of shape (n_samples_Y, )
            each row is a sequence (string)
        l : int, default 3
            number of l-mers (length of 'word')
        s_l: int, default 1
            number of shifting allowed
            # TODO: confirm whether we want to use the choice in the paper
        Returns
        -------
        kernel_matrix : array of shape (n_samples_X, n_samples_Y)
        """

        if type(X[0,]) is not str and type(X[0,]) is not np.str_: 
            X = self.inverse_label(X)  

        if Y is None:
            Y = X
        elif type(Y[0,]) is not str  and type(Y[0,]) is not np.str_:
            Y = self.inverse_label(Y)

        K = np.zeros((X.shape[0], Y.shape[0]))
        # assume all seq has the same total length
        L = len(X[0])

        assert len(self.l_list) == 1
        l = self.l_list[0]

        for d in range(1, l+1):
            for j in range(0, L - d + 1):
                for s in range(0, self.s +1):
                    if s + j <= L:
                        beta = 2 * float(l - d + 1)/float(l ** 2 + l)
                        delta = 1.0/(2 * (s + 1))
                        K += beta * delta * \
                            (self.dotproduct_phi(X, Y, l_list=[d], j_X=j+s, j_Y=j, d=d) + 
                            self.dotproduct_phi(X, Y, l_list=[d], j_X=j, j_Y=j+s, d=d))

        if plot_flag:
            self.plot_kernel({'K': K}, title = 'WD Kernel with Shift Matrix')
        if eval_gradient:
            if not self.hyperparameter_sigma_0.fixed:
                K_gradient = np.empty((K.shape[0], K.shape[1], 1))
                K_gradient[..., 0] = 2 * self.sigma_0 ** 2
                return K, K_gradient
            else:
                return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K

        
    def __call__(self, X, Y=None, eval_gradient=False, print_flag = False, plot_flag = False):
        """Weighted degree kernel with shifts.
        Compute the mixed spectrum kernel between X and Y:
            K(x, y) = \sum_{d = 1}^{l} \sum_j^{L-d} \sum_{s=0 and s+j <= L}
                beta_d * gamma_j * delta_s *
                (k_d^{spectrum}(x[j+s:j+s+d],y[j:j+d]) + k_d^{spectrum}(x[j:j+d],y[j+s:j+s+d]))
        for each pair of rows x in X and y in Y.
        when Y is None, Y is set to be equal to X.

        beta_d = 2 frac{l - d + 1}{l^2 + 1}
        gamma_j = 1
        delta_s = 1/(2(s+1))

        TODO: to confirm why shift useful?
        
        Parameters
        ----------
        X : array of shape (n_samples_X, )
            each row is a sequence (string)
        Y : array of shape (n_samples_Y, )
            each row is a sequence (string)
        l : int, default 3
            number of l-mers (length of 'word')
        s_l: int, default 1
            number of shifting allowed
            # TODO: confirm whether we want to use the choice in the paper
        Returns
        -------
        kernel_matrix : array of shape (n_samples_X, n_samples_Y)
        """
        if Y is None:
            Y = X

        if len(X) == self.n_train and len(Y) == self.n_train: # K(train, train)
            return self.kernel_all_normalised[:self.n_train, :self.n_train]
        elif len(X) == self.n_test and len(Y) == self.n_test: # K(test, test)
            return self.kernel_all_normalised[-self.n_test:, -self.n_test:]
        elif len(X) == self.n_train and len(Y) == self.n_test: # K(train, test)
            return self.kernel_all_normalised[:self.n_train, -self.n_test:]
        elif len(X) == self.n_test and len(Y) == self.n_train: # K(test, train)
            return self.kernel_all_normalised[-self.n_test:, :self.n_train]
        else:
            raise ValueError('Cannot slice a kernel matrix.')

