import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn import preprocessing
import matplotlib.pyplot as plt

# To be able to normalise the kernel matrix
# We cannot use Pairwise kernel for GP model
# So we will inherit from sklearn gaussain process kernel class
# https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/gaussian_process/kernels.py#L1213

from sklearn.gaussian_process.kernels import Kernel, Hyperparameter

BASES = ['A','C','G','T']


def Phi(X, Y, l_list = [3], j_X=0, j_Y=0, d=None, weight_flag= False, padding_flag = False, gap_flag = False):
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
        
        # centering
        
        embedded_center_X = np.nanmean(embedded_X, axis = 0)
        embedded_X -= embedded_center_X 

        embedded_center_Y = np.nanmean(embedded_Y, axis = 0)
        embedded_Y -= embedded_center_Y 

        # unit norm 
        normalised_embedded_X = preprocessing.normalize(embedded_X, norm = 'l2')
        normalised_embedded_Y = preprocessing.normalize(embedded_Y, norm = 'l2')

        if weight_flag:
            feature_names = cv.get_feature_names()
            W = weights_for_Phi(feature_names)
            return normalised_embedded_X, normalised_embedded_Y, W
        else:
            return normalised_embedded_X, normalised_embedded_Y
        
        #return embedded_X, embedded_Y


def generate_gapped_kmer(sequence, l):
    words_gapped = []
    for a in range(len(sequence) - l + 1):
        # Only works for l = 3 for now
        words_gapped.append(sequence[a] + 'N' + sequence[a+l-1]) 
    return words_gapped

def inverse_label(X):
    """convert_to_string
    """
    le = preprocessing.LabelEncoder()
    le.fit(BASES)

    inversed_label_X = np.array(["".join(le.inverse_transform(list(X[i])))\
                                     for i in range(X.shape[0])])
    
    return inversed_label_X

def weights_for_Phi(feature_names):
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
                
class Spectrum_Kernel(Kernel):
    """
    
    Parameters
    ------------------
    l : list, default [3]
            list of number of l-mers (length of 'word')
            For example [2,3] means extracting all substrings of length 2 and 3 
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
    def __init__(self, l_list=[3], weight_flag = False, padding_flag = False, gap_flag = False,
                 sigma_0=1e-10, sigma_0_bounds=(1e-10,1e10)):
        self.l_list = l_list
        self.weight_flag = weight_flag
        self.padding_flag = padding_flag
        self.gap_flag = gap_flag
        self.sigma_0 = sigma_0
        self.sigma_0_bounds = sigma_0_bounds

    @property
    def hyperparameter_sigma_0(self):
        return Hyperparameter("sigma_0", "numeric", self.sigma_0_bounds)
    
    def __call__(self, X, Y=None, eval_gradient=False, print_flag = False, plot_flag = False):
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
            phi_X, phi_Y, W = Phi(X, Y, self.l_list, weight_flag = self.weight_flag,
                                  padding_flag=self.padding_flag, gap_flag=self.gap_flag)
            K = np.normalisation(phi_X.dot(W).dot(phi_Y.T)) + self.sigma_0 ** 2
        else:
            phi_X, phi_Y = Phi(X, Y, self.l_list, weight_flag = self.weight_flag, 
                                padding_flag=self.padding_flag, gap_flag=self.gap_flag)
            K = phi_X.dot(phi_Y.T) + self.sigma_0 ** 2

        #K = self.normalisation(K)

        if plot_flag:
            self.plot_kernel({'K': K})

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
        Compute the distance between X and Y based on spectrum kernel:
            d_{l}^{spectrum}(x, y) = sqrt(||phi(x) - phi(y)||^2)
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

        phi_X, phi_Y = Phi(X, Y, self.l_list, weight_flag = self.weight_flag, 
                            padding_flag=self.padding_flag, gap_flag=self.gap_flag)

        distance_matrix = np.zeros((phi_X.shape[0], phi_Y.shape[0]))

        for i in range(phi_X.shape[0]):
            for j in range(phi_Y.shape[0]):
                if j >= i:
                    #distance_matrix[i,j] = np.sqrt(np.sum(np.power((phi_X[i,:]- phi_Y[j,:]), 2)))
                    distance_matrix[i,j] = np.linalg.norm(x = (phi_X[i,:] - phi_Y[j,:]), ord= 2)

        for i in range(phi_X.shape[0]):
            for j in range(phi_Y.shape[0]):
                if j < i:
                    distance_matrix[i,j] = distance_matrix[j,i]

        return  distance_matrix, phi_X, phi_Y
        #return phi_X, phi_Y
    
    def normalisation(self, kernel):
        spherical_kernel = np.zeros_like(kernel)
        d = min(kernel.shape[0], kernel.shape[1])
        for i in range(d):
            for j in range(d):
                spherical_kernel[i,j] = kernel[i,j]/np.sqrt(kernel[i,i] * kernel[j,j])

        kernel = spherical_kernel

        standardized_kernel = np.zeros_like(kernel)
        kernel_mean = np.mean(kernel, axis = (0,1))
        n = kernel.shape[0]
        kernel_trace = np.trace(kernel)

        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                standardized_kernel[i,j] = kernel[i,j]/(1.0/n * kernel_trace  - kernel_mean)
        
        return standardized_kernel                

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

    def plot_kernel(self, K_dict):
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
                
        plt.title('Spectrum Kernel Matrix')
        plt.show()

class Sum_Spectrum_Kernel(Spectrum_Kernel):

    def __init__(self, l_list=[3], b = 0.33, weight_flag = False, padding_flag = False, gap_flag = False,
                 sigma_0=1e-10, sigma_0_bounds=(1e-10,1e10)):
        """
        b: float
            the weight for K_B
        """
        super().__init__(l_list, weight_flag, padding_flag, gap_flag, sigma_0, sigma_0_bounds)
        self.b = b

    def __call__(self, X, Y=None, eval_gradient=False, print_flag = False, plot_flag = False):
        """
        Compute the spectrum kernel between X and Y:
            k_{l}^{spectrum}(x, y) = <phi(x), phi(y)>
        for each pair of rows x in X and y in Y.
        when Y is None, Y is set to be equal to X.

        Parameters
        ----------
        X : array of shape (n_samples_X, )
            each row is a sequence (string)
        Y : array of shape (n_samples_Y, )
            each row is a sequence (string)
        l : int, default 3
            number of l-mers (length of 'word')
        j_X : int
            start position of sequence in X
        j_Y : int
            start position of sequence in Y
        d : int, default None
            if None, set to the length of sequence
            d is the length of analysed sequence
            j + d is end position of sequence 
        Returns
        -------
        kernel_matrix : array of shape (n_samples_X, n_samples_Y)
        """

        # if each row is not a string, convert it to string
        if type(X[0,]) is not str and type(X[0,]) is not np.str_: 
            X = inverse_label(X)           

        if Y is None:
            Y = X
        elif type(Y[0,]) is not str  and type(Y[0,]) is not np.str_:
            Y = inverse_label(Y)

        

        # split each string into three parts A [:7] + B [7:13] + C [13:]
        # the reason we consider sum of kernel is that 
        # each part has high correlations to itself, but less with the other two
        
        X_A, X_B, X_C = self.split(X)
        Y_A, Y_B, Y_C = self.split(Y)

        if print_flag:
            print('X_A: ', X_A)
            print('Y_A: ', Y_A)
            print('X_B: ', X_B)
            print('Y_B: ', Y_B)
            print('X_C: ', X_C)
            print('Y_C: ', Y_C)

        if self.weight_flag:
            phi_X_A, phi_Y_A, W_A = Phi(X_A, Y_A, self.l_list, weight_flag = self.weight_flag,
                                  padding_flag=self.padding_flag, gap_flag=self.gap_flag)
            phi_X_B, phi_Y_B, W_B = Phi(X_B, Y_B, self.l_list, weight_flag = self.weight_flag,
                                  padding_flag=self.padding_flag, gap_flag=self.gap_flag)
            phi_X_C, phi_Y_C, W_C = Phi(X_C, Y_C, self.l_list, weight_flag = self.weight_flag,
                                  padding_flag=self.padding_flag, gap_flag=self.gap_flag)
            
            K_A = self.normalisation(phi_X_A.dot(W_A).dot(phi_Y_A.T))
            K_B = self.normalisation(phi_X_B.dot(W_B).dot(phi_Y_B.T))
            K_C = self.normalisation(phi_X_C.dot(W_C).dot(phi_Y_C.T))
        else:
            phi_X_A, phi_Y_A = Phi(X_A, Y_A, self.l_list, weight_flag = self.weight_flag,
                                  padding_flag=self.padding_flag, gap_flag=self.gap_flag)
            phi_X_B, phi_Y_B = Phi(X_B, Y_B, self.l_list, weight_flag = self.weight_flag,
                                  padding_flag=self.padding_flag, gap_flag=self.gap_flag)
            phi_X_C, phi_Y_C = Phi(X_C, Y_C, self.l_list, weight_flag = self.weight_flag,
                                  padding_flag=self.padding_flag, gap_flag=self.gap_flag)
            
            K_A = phi_X_A.dot(phi_Y_A.T)
            K_B = phi_X_B.dot(phi_Y_B.T)
            K_C = phi_X_C.dot(phi_Y_C.T)
        #K_A = self.normalisation(K_A)
        #K_B = self.normalisation(K_B)
        #K_C = self.normalisation(K_C)

        K = (1-self.b)/2.0 * K_A + self.b * K_B + (1-self.b)/2.0* K_C + self.sigma_0 ** 2

        kernel_matrix = {'K_A': K_A,
                        'K_B': K_B, 
                        'K_C': K_C, 
                        'K': K,  
        
        }

        if plot_flag:
            self.plot_kernel(kernel_matrix)

        """
        print('A', phi_X_A.dot(phi_Y_A.T))
        print()
        print('B', phi_X_B.dot(phi_Y_B.T))
        print()
        print('C', phi_X_C.dot(phi_Y_C.T))
        
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

    def split(self, X, split_idx = [7, 13]):
        """split each string into three parts A [:7] + B [7:13] + C [13:]
        """
        A = []
        B = []
        C = []
        for i in X:
            A.append(i[:split_idx[0]])
            B.append(i[split_idx[0]: split_idx[-1]])
            C.append(i[split_idx[-1]:])
        return np.asarray(A), np.asarray(B), np.asarray(C)  

    
# -------------------------------------------------------------------------

def mixed_spectrum_kernel(X, Y=None, l = 3):
    """
    Compute the mixed spectrum kernel between X and Y:
        K(x, y) = \sum_{d = 1}^{l} beta_d k_d^{spectrum}(x,y)
    for each pair of rows x in X and y in Y.
    when Y is None, Y is set to be equal to X.

    beta_d = 2 frac{l - d + 1}{l^2 + 1}

    Parameters
    ----------
    X : array of shape (n_samples_X, )
        each row is a sequence (string)
    Y : array of shape (n_samples_Y, )
        each row is a sequence (string)
    l : int, default 3
        number of l-mers (length of 'word')
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X
    K = np.zeros((X.shape[0], Y.shape[0]))
    for d in range(1, l+1):
        #print(d)
        beta = 2 * float(l - d + 1)/float(l ** 2 + 1)
        K += beta * spectrum_kernel(X, Y, l = d)
    return K

def WD_kernel(X, Y=None, l = 3):
    """Weighted degree kernel.
    Compute the mixed spectrum kernel between X and Y:
        K(x, y) = \sum_{d = 1}^{l} \sum_j^{L-d} 
            beta_d k_d^{spectrum}(x[j:j+d],y[j:j+d])
    for each pair of rows x in X and y in Y.
    when Y is None, Y is set to be equal to X.

    beta_d = 2 frac{l - d + 1}{l^2 + 1}

    Parameters
    ----------
    X : array of shape (n_samples_X, )
        each row is a sequence (string)
    Y : array of shape (n_samples_Y, )
        each row is a sequence (string)
    l : int, default 3
        number of l-mers (length of 'word')
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X
    K = np.zeros((X.shape[0], Y.shape[0]))
    # assume all seq has the same total length
    L = len(X[0])

    for d in range(1, l+1):
        #print(d)
        for j in range(0, L - d + 1):
            beta = 2 * float(l - d + 1)/float(l ** 2 + 1)
            K += beta * spectrum_kernel(X, Y, d, j, j, d)
    return K

def WD_shift_kernel(X, Y=None, l = 3, shift_range = 1):
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
    shift_range: int, default 1
        number of shifting allowed
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X
    K = np.zeros((X.shape[0], Y.shape[0]))
    
    L = len(X[0]) # assume all seq has the same total length

    for d in range(1, l+1):
        #print(d)
        for j in range(0, L - d + 1):
            for s in range(shift_range+1): # range is right open
                if s + j <= L:
                    beta = 2 * float(l - d + 1)/float(l ** 2 + 1)
                    delta = 1.0/(2 * (s + 1))
                    K += beta * delta * (spectrum_kernel(X, Y, d, j+s, j, d) + spectrum_kernel(X, Y, d, j, j+s, d))
    return K



#Test example
#spec_kernel = Spectrum_Kernel()
#spec_kernel.__call__(np.array(['ACTGAC', 'ACTTTT']), np.array(['ACTGAC', 'ACTTTT']))
Phi(np.array(['ACTGAC', 'ACTTTT']), np.array(['ACTGAC', 'ACTTTT']), [3])
