import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
import matplotlib.pyplot as plt

from codes.embedding import Embedding
from codes.environment import Rewards_env

import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pickle
import codes.config as config # provide configuration, e.g. global variable, path

# Aug 2020 Mengyan Zhang 
# String Kernel Classes, taking strings as input
# including spectrum kernel, weighted degree kernel and weighted degree kernel with shift 
# The implementation is based on Support Vector Machines and Kernels for Computational Biology 
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2547983/
# Inherit from sklearn gaussain process kernel class
# https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/gaussian_process/kernels.py#L1213

# When creating an instance, the whole (train + test) kernel matrix is pre-calculated and normalised
# For Gaussian Process Regression, the specific kernel matrix is sliced from the calculated kernel

# The all bases for input strings, we use the RBS sequences here


# global features for all known and design data
# if os.path.exists('../../data/unknown_design.csv'):
#     unknown_df = pd.read_csv('../../data/unknown_design.csv') # sequences with unknown label
#     unknown_features = Rewards_env(np.asarray(unknown_df[['RBS', 'RBS6']]), 'label').embedded
# else:
#     unknown_features = None
#     print('No default features for kernel instance. Please specify features.')

class String_Kernel(Kernel):
    """Base class for string kernels
    
    Parameters
    ------------------
    * For initialisation, calculating the whole kernel

    features: array
        list of features for all training and testing data
    n_train: int
        number of training data
    n_test: int 
        number of testing data
    kernel_all: (n_train+n_test, n_train+n_test)
        kernel matrix for all data
    kernel_all_normalised: (n_train+n_test, n_train+n_test)
        kernel matrix for all data with centering and unit norm

    * Kernel hyperparameters

    l : int, default 3
        number of l-mers (length of 'word')
    # TODO: delta weight flag
    padding_flag: Boolean, default False
        indicates whether adding padding characters before and after sequences
    gap_flag: Boolean, default False
        indicates whether generates substrings with gap

    # TODO: change hyperparameters maybe (now set to the same as Dotproduct)
    sigma_0 : float >= 0, default: 0.0
        signal std
        Parameter controlling the inhomogenity of the kernel. If sigma_0=0,
        the kernel is homogenous.
    sigma_0_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on l
    """
    
    INIT_FLAG = False
    KERNEL_ALL_NORM = None
    DISTANCE_ALL = None

    def __init__(self, l=6, 
                 padding_flag = False, gap_flag = False,
                 sigma_0= 1, #, sigma_0_bounds=(1e-10,1e10)):
                 lengthscale_rate = None,
                 kernel_norm_flag = True,
                 centering_flag = True,
                 unit_norm_flag = True,
                ):
        
        self.l = l
        self.padding_flag = padding_flag
        self.gap_flag = gap_flag
        self.sigma_0 = sigma_0
        self.lengthscale_rate = lengthscale_rate

        self.kernel_norm_flag = kernel_norm_flag
        self.centering_flag = centering_flag
        self.unit_norm_flag = unit_norm_flag

        # self.sigma_0_bounds = sigma_0_bounds

        # calculate the whole kernel matrix for the concatenate train and test data with normalisation
        # then slice the corresponding part for GPR

        # train n_train * d
        # test n_test * d
        # features (n_train + n_test) * d
        # kernel_all (n_train + n_test) * (n_train + n_test)
        
        # self.train_idx = np.asarray(train_idx)
        # self.test_idx = np.asarray(test_idx)
        # self.features = features # train has to be put at the front
        # print('train size: ', self.n_train)
        # print('test size: ', self.n_test)
        # print('feature size: ', self.features.shape)
        # assert self.n_train + self.n_test == self.features.shape[0]

        #----------------------------------------------------------------------------------------
        # reconstructure: use all 4138 unique seqs for kernel normalisation 
        if os.path.exists(config.SAVED_KERNEL_PATH):
            with open(config.SAVED_KERNEL_PATH, 'rb') as handle:
                SAVED_KERNELS = pickle.load(handle)
                # print('available kernels: ', list(SAVED_KERNELS.keys()))
        else:
            SAVED_KERNELS = {} # key: kernel name (with parameters); value: kernel matrix
        print(self.kernel_name_para)
        if self.kernel_name_para in list(SAVED_KERNELS.keys()):
            print('Load saved kernel matrix...')
            self.kernel_all_normalised = SAVED_KERNELS[self.kernel_name_para]
            # TODO: need to check whether the saved kernel is generated with same parameters
        else:
            print('No saved kernel matrix. Calculate and save kernel matrix...')
            # seq = np.load(SAVED_IDX_SEQ_PATH)['seqList']
            with open(config.SAVED_IDX_SEQ_PATH, 'rb') as handle:
                seq = pickle.load(handle)['seq_list']
            features = Rewards_env(np.asarray(seq).reshape(len(seq), 1), 'label').embedded

            self.kernel_all = self.cal_kernel(features, features)
            assert (self.kernel_all).all() == (self.kernel_all.T).all() # check symmetric

            if kernel_norm_flag:
                self.kernel_all_normalised = self.normalisation(self.kernel_all, centering_flag, unit_norm_flag) * sigma_0
            else:
                # print('USE non-normalised kernel!')
                self.kernel_all_normalised = self.kernel_all * sigma_0
            SAVED_KERNELS[self.kernel_name_para] = self.kernel_all_normalised
            with open(config.SAVED_KERNEL_PATH, 'wb') as handle:
                pickle.dump(SAVED_KERNELS, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Kernel saved.')
        #----------------------------------------------------------------------------------------

        # if not type(self).INIT_FLAG: # to avoid init again in clone (deepcopy) in GPR.fit
        #     if self.features.shape[0] < unknown_features.shape[0]:
        #         # use all known + unknown seq for normalisation
        #         self.known_unknown_features = np.concatenate((self.features, unknown_features), axis=0)
        #         print('add unknown seq: ', self.known_unknown_features.shape)
                
        #     self.kernel_all = self.cal_kernel(self.known_unknown_features, self.known_unknown_features)
        #     assert (self.kernel_all).all() == (self.kernel_all.T).all() # check symmetric

        #     # check positive definite
        #     # print('eignh for kernel_all:')
        #     # print(np.linalg.eigh(self.kernel_all))
        #     # L = np.linalg.cholesky(self.kernel_all)
        #     # print('kernel_all is positive definite')

        #     # TODO: decide whether to use sigma_0
        #     if kernel_norm_flag:
        #         self.kernel_all_normalised = self.normalisation(self.kernel_all, centering_flag, unit_norm_flag) * sigma_0
        #     else:
        #         print('USE non-normalised kernel!')
        #         self.kernel_all_normalised = self.kernel_all * sigma_0

        #     # covert back to only train and test data
        #     self.kernel_all_normalised = self.kernel_all_normalised[:self.n_train+self.n_test, :self.n_train+self.n_test]
        #     print('back to normalised: ', self.kernel_all_normalised.shape)
    
        #     #print(np.linalg.eigh(self.kernel_all_normalised)[0])
            
        #     # check positive definite
        #     # print('eignh for kernel_all_normalised:')
        #     # print(np.linalg.eigh(self.kernel_all_normalised))
        #     # L = np.linalg.cholesky(self.kernel_all_normalised)
        #     # print('kernel_all_normalised is positive definite')
        #     self.distance_all = self.distance(self.kernel_all_normalised)

        #     # Try idea to introduce lengthscale into wd kernel
        #     # Take the WD kernel and calculate the squared distance matrix corresponding to it. 
        #     # For this squared distance matrix, you can compute the Gaussian kernel using the formula exp(-D^2/(length scale))
        #     if self.lengthscale_rate != None:
        #         # print(self.distance_all**2)
        #         # plt.hist((self.distance_all**2)[np.triu_indices(self.distance_all.shape[0], self.distance_all.shape[1])])
        #         # plt.title('Histogram of upper triangle of D^2')
        #         lengthscale = np.quantile(self.distance_all**2, self.lengthscale_rate)
        #         print('lengthscale: ', lengthscale)
        #         self.kernel_all_normalised = self.normalisation(- self.distance_all**2/lengthscale, centering_flag, unit_norm_flag) * sigma_0
        #     print('init kernel')
            
        #     type(self).INIT_FLAG = True
        #     type(self).KERNEL_ALL_NORM = self.kernel_all_normalised
        #     type(self).DISTANCE_ALL = self.distance_all

        # else:
        #     self.kernel_all_normalised = type(self).KERNEL_ALL_NORM
        #     self.distance_all = type(self).DISTANCE_ALL

    # @property
    # def hyperparameter_sigma_0(self):
    #     return Hyperparameter("sigma_0", "numeric", self.sigma_0_bounds)
    
    def cal_kernel(self, X, Y=None):
        """Calculate K(X,Y)
        """

    def distance(self, K):
        """Compute the distance based on given kernel K:
        d(x,y) = sqrt(k(x,x) + k(y,y) - 2k(x,y))
        """
        K_diag = K.diagonal().copy()

        K1 = np.zeros((len(K_diag), len(K_diag)))
        K2 = np.zeros((len(K_diag), len(K_diag)))

        for i, diag in enumerate(K_diag):
            K1[i,:] = diag
            K2[:,i] = diag

        distance_matrix = np.sqrt(K1+K2-2*K)
        return  distance_matrix
    
    def __call__(self, X, Y=None, eval_gradient=False, print_flag = False, plot_flag = False):
        """ Slide kernels. 
        Judge the input by the shape of X,Y and the indicated n_train, n_test.
        """

        # query X, Y index list

        # X_index_list = []
        # for i in range(X.shape[0]):
        #     inverse_list = self.encoder.inverse_transform(X[i,:])
        #     X_index_list.append(''.join(j) for j in inverse_list)
        # if Y is None:
        #     Y_index_list = X_index_list
        # else:
        #     Y_index_list
        #     for i in range(Y.shape[0]):
        #         inverse_list = self.encoder.inverse_transform(Y[i,:])
        #         Y_index_list.append(''.join(j) for j in inverse_list)

        with open(config.SAVED_IDX_SEQ_PATH, 'rb') as handle:
            saved_idx_seq = pickle.load(handle)
        idx_seq_dict = saved_idx_seq['idx_seq_dict']

        X_index_list = []
        for i in self.inverse_label(X):
            X_index_list.append(idx_seq_dict[i])

        if Y is None:
            Y_index_list = X_index_list
        else:
            Y_index_list = []
            for i in self.inverse_label(Y):
                Y_index_list.append(idx_seq_dict[i])

        # create kernel matrix using saved kernel_all_normalised
        kernel = np.zeros((len(X_index_list), len(Y_index_list)))
        
        for i, xidx in enumerate(X_index_list):
            for j, yidx in enumerate(Y_index_list):
                #if i <= j:
                kernel[i, j] = self.kernel_all_normalised[xidx,yidx]
        # kernel = kernel + kernel.T - np.diag(np.diag(kernel))

        # if len(X) == self.n_train and len(Y) == self.n_train: # K(train, train)
        #     K = self.kernel_all_normalised[:self.n_train, :self.n_train].copy()
        # elif len(X) == self.n_test and len(Y) == self.n_test: # K(test, test)
        #     K = self.kernel_all_normalised[-self.n_test:, -self.n_test:].copy()
        # elif len(X) == self.n_train and len(Y) == self.n_test: # K(train, test)
        #     K = self.kernel_all_normalised[:self.n_train, -self.n_test:].copy()
        # elif len(X) == self.n_test and len(Y) == self.n_train: # K(test, train)
        #     K = self.kernel_all_normalised[-self.n_test:, :self.n_train].copy()
        # else:
        #     raise ValueError('Cannot slice a kernel matrix.')
        # plt.imshow(kernel)
        # plt.colorbar()
        # plt.show()
        if eval_gradient:
            # if not self.hyperparameter_sigma_0.fixed:
            #     K_gradient = np.empty((K.shape[0], K.shape[1], 1))
            #     K_gradient[..., 0] = 2 * self.sigma_0 ** 2
            #     return K, K_gradient
            # else:
            return kernel, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return kernel
    
    def normalisation(self, kernel, centering_flag = True, unit_norm_flag = True):
        # https://jmlr.csail.mit.edu/papers/volume12/kloft11a/kloft11a.pdf
        # Sec 4.2.2
        # First calculate zero mean kernel
        # Then calculate unit variance
        # The the variance (defined as the 1/n trace - kernel mean) is 1 
        # normalise over the whole matrix (train, test)
        print('centering_flag: ', centering_flag)
        print('unit_norm_flag: ', unit_norm_flag)

        if centering_flag:
        
            # centering
            one_mat = np.ones(kernel.shape) # kernel shape, i.e. (n_train+n_test, n_train+n_test)
            one_vec = np.ones((kernel.shape[0],1)) # (n_train+n_test, 1)

            row_sum = np.mean(kernel,axis=0).T 
            row_sum = row_sum.reshape(row_sum.shape[0],1) # (n_train+n_test, 1)
        
            kernel_centered = kernel - row_sum.dot(one_vec.T) - one_vec.dot(row_sum.T) +\
                np.mean(row_sum)*one_mat

            kernel = kernel_centered

        # kernel_centered += 1e-10 * np.identity(kernel_centered.shape[0])
        # print('eignh for kernel_centered:')
        # print(np.linalg.eigh(kernel_centered))
        # L = np.linalg.cholesky(kernel_centered)
        # print('kernel centered is positive definite')

        if unit_norm_flag:

            # unit variance
            
            Kii = np.diag(kernel)
            Kii.shape = (len(Kii),1)
            kernel_unit_norm =  np.divide(kernel, np.sqrt(Kii*Kii.T))

            kernel = kernel_unit_norm

        return kernel  
        # return kernel_centered

    def generate_sentences(self, X, l, j_X, d):
        """generate sentences for a dataset of sequences/strings
        """
        sentences = []

        for i in range(X.shape[0]):
            words = []
            sequence= X[i][j_X:j_X + d]

            if self.padding_flag:
                sequence = 'ZZ' + sequence + 'ZZ' # Padding_flag
                #sequence = sequence[-2:] + sequence + sequence[:2] # Padding_flag
            
            words += [sequence[a:a+l] for a in range(len(sequence) - l + 1)]

            if self.gap_flag:
                words_gapped = generate_gapped_kmer(sequence, l)
                words = words + words_gapped

            sentence = ' '.join(words)
            sentences.append(sentence)

        return sentences

    def Phi(self, X, Y, l = 3, j_X=0, j_Y=0, d=None):
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
        l : default 3
            number of l-mers (length of 'word')
        j_X : int
            start position of sequence in X
        j_Y : int
            start position of sequence in Y
        d : int
            the length of analysed sequence
            j + d is end position of sequence 
        padding_flag: Boolean, default False
            indicates whether adding padding characters before and after sequences
        gap_flag: Boolean, default False
            indicates whether generates substrings with gap

        Returns
        ----------------------------------------------------
        embedded matrix of X: array
            num_X * num_embedded_features
        embedded matrix of Y
            num_Y * num_embedded_features
        """
        if d is None:
            d = len(X[0]) 

        sentences_X = self.generate_sentences(X, l, j_X, d)
        sentences_Y = self.generate_sentences(Y, l, j_Y, d)

        cv = CountVectorizer(analyzer='word',token_pattern=u"(?u)\\b\\w+\\b")
        #cv =  CountVectorizer()

        embedded = cv.fit_transform(sentences_X + sentences_Y).toarray()
        embedded_X = embedded[: X.shape[0], :].astype(float)
        embedded_Y = embedded[-Y.shape[0]: , :].astype(float)
        
        return embedded_X, embedded_Y

    def dotproduct_phi(self, X, Y, l, j_X, j_Y, d):
        phi_X, phi_Y = self.Phi(X, Y, l, j_X, j_Y, d)
        kernel = phi_X.dot(phi_Y.T) # + self.sigma_0 ** 2

        # kernel += 1e-20 * np.identity(kernel.shape[0])

        # check positive definite
        # L = np.linalg.cholesky(kernel)
        # print('kernel inside of wd is positive definite')
        # print('eignh for kernel inside of wd:')
        # print(np.linalg.eigh(kernel))
        return kernel

    def inverse_label(self, X):
        """convert features to string
        """
        if type(X[0]) is str:
            return X    
        else:
            le = preprocessing.LabelEncoder()
            le.fit(config.BASES)

            inversed_label_X = np.array(["".join(le.inverse_transform(list(X[i])))\
                                            for i in range(X.shape[0])])
            
            return inversed_label_X

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X). For GPR.
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

    # def __repr__(self):
    #     return "{0}(sigma_0={1:.3g})".format(
    #         self.__class__.__name__, self.sigma_0)

    #--------------------------------------------------------------------------
    # TODO: need to tidy up

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
    """Weighted degree kernel with shift.

    Compute the mixed spectrum kernel between X and Y:
            K(x, y) = \sum_{d = 1}^{l} \sum_j^{L-d} \sum_{s=0 and s+j <= L}
                beta_d * gamma_j * delta_s *
                (k_d^{spectrum}(x[j+s:j+s+d],y[j:j+d]) + k_d^{spectrum}(x[j:j+d],y[j+s:j+s+d]))
    for each pair of rows x in X and y in Y.
    when Y is None, Y is set to be equal to X.

    beta_d = 2 frac{l - d + 1}{l^2 + 1}
    gamma_j = 1
    delta_s = 1/(2(s+1))
    """

    def __init__(self, l=6, 
                 padding_flag = False, gap_flag = False,
                 sigma_0= 1, #, sigma_0_bounds=(1e-10,1e10)):
                 lengthscale_rate = None,
                 kernel_norm_flag = True,
                 centering_flag = True,
                 unit_norm_flag = True,
                 s = 0):
        """

        Paramters
        ---------------------------------------
        s: int
            shift length. When s=0, the same as weighted degree kernel.
        """
        self.s = s
        self.kernel_name_para = 'wds_l'+str(l)+'_sigma0_'+ str(sigma_0) + '_s' + str(s) + '_center_' + str(centering_flag) + '_unitnorm_' + str(unit_norm_flag)
        super().__init__(l, padding_flag, gap_flag, sigma_0, lengthscale_rate, 
                kernel_norm_flag, centering_flag, unit_norm_flag) 
                #sigma_0, sigma_0_bounds)

    def cal_kernel(self, X, Y=None, eval_gradient=False, print_flag = False, plot_flag = False):
        """Weighted degree kernel with shifts. Calculate the whole kernel.
        
        Parameters
        ----------
        X : array of shape (n_samples_X, )
            each row is a sequence (string)
        Y : array of shape (n_samples_Y, )
            each row is a sequence (string)
        
        Returns
        -------
        kernel_matrix : array of shape (n_samples_X, n_samples_Y)
        """

        # Format checking
        if type(X[0,]) is not str and type(X[0,]) is not np.str_: 
            X = self.inverse_label(X)  

        if Y is None:
            Y = X
        elif type(Y[0,]) is not str  and type(Y[0,]) is not np.str_:
            Y = self.inverse_label(Y)

        K = np.zeros((X.shape[0], Y.shape[0]))

        # assume all seq has the same total length
        L = len(X[0])

        for d in range(1, self.l+1):
            for j in range(0, L - d + 1):
                for s in range(0, self.s +1):
                    if s + j <= L:
                        beta = 2 * float(self.l - d + 1)/float(self.l ** 2 + self.l)
                        delta = 1.0/(2 * (s + 1))
                        # Todo: adding gamma? RASE: recognition of alternatively spliced exons in C.elegans.
                        gamma = 1.0/L
                        K += beta * delta * gamma * \
                            (self.dotproduct_phi(X, Y, l=d, j_X=j+s, j_Y=j, d=d) + 
                            self.dotproduct_phi(X, Y, l=d, j_X=j, j_Y=j+s, d=d))

        #K += self.sigma_0 ** 2
        
        if plot_flag:
            self.plot_kernel({'K': K}, title = 'WD Kernel with Shift Matrix')
        # if eval_gradient:
        #     if not self.hyperparameter_sigma_0.fixed:
        #         K_gradient = np.empty((K.shape[0], K.shape[1], 1))
        #         K_gradient[..., 0] = 2 * self.sigma_0 ** 2
        #         return K, K_gradient
        #     else:
        #         return K, np.empty((X.shape[0], X.shape[0], 0))
        # else:
        return K

class Spectrum_Kernel(String_Kernel):
    """Spectrum Kernel
    # TODO: REWRITE
    """

    def cal_kernel(self, X, Y=None, eval_gradient=False, print_flag = False, plot_flag = False):
        """Spectrum Kernel. Calculate the whole kernel.
        
        Parameters
        ----------
        X : array of shape (n_samples_X, )
            each row is a sequence (string)
        Y : array of shape (n_samples_Y, )
            each row is a sequence (string)
        
        Returns
        -------
        kernel_matrix : array of shape (n_samples_X, n_samples_Y)
        """

        # Format checking
        if type(X[0,]) is not str and type(X[0,]) is not np.str_: 
            X = self.inverse_label(X)  

        if Y is None:
            Y = X
        elif type(Y[0,]) is not str  and type(Y[0,]) is not np.str_:
            Y = self.inverse_label(Y)

        K = self.dotproduct_phi(X, Y, l=self.l)

        return K

# TODO: implement other kernels