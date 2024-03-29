import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn import preprocessing

def Phi(X, Y, l, j_X, j_Y, d):
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
    l : int, default 3
        number of l-mers (length of 'word') 
    j_X : int
        start position of sequence in X
    j_Y : int
        start position of sequence in Y
    d : int
        the length of analysed sequence
        j + d is end position of sequence 

    Returns
    ----------------------------------------------------
    embedded matrix of X: array
        num_X * num_embedded_features
    embedded matrix of Y
        num_Y * num_embedded_features
    """
    num_X = X.shape[0]
    num_Y = Y.shape[0]

    sentences = []

    for i in range(num_X):
        sequence= X[i][j_X:j_X + d]
        words = [sequence[a:a+l] for a in range(len(sequence) - l + 1)]
        sentence = ' '.join(words)
        sentences.append(sentence)

    for i in range(num_Y):
        sequence= Y[i][j_Y:j_Y + d]
        words = [sequence[a:a+l] for a in range(len(sequence) - l + 1)]
        sentence = ' '.join(words)
        sentences.append(sentence)
    #print(sentences)
    cv = CountVectorizer(analyzer='word',token_pattern=u"(?u)\\b\\w+\\b")
    #cv =  CountVectorizer()
    embedded = cv.fit_transform(sentences).toarray()
    #print(embedded)
    #print(cv.get_feature_names())
    normalised_embedded = preprocessing.normalize(embedded, norm = 'l2')
    #print(normalised_embedded)

    return normalised_embedded[: num_X, :], normalised_embedded[-num_Y: , :]


def spectrum_kernel(X, Y=None, l = 3, j_X = 0, j_Y = 0, d = None):
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
    if Y is None:
        Y = X
    if d is None:
        d = len(X[0]) 
    # sequence cannot pass the check 
    # X, Y = check_pairwise_arrays(X, Y)
    phi_X, phi_Y = Phi(X, Y, l, j_X, j_Y, d)
    print(phi_X)
    print(phi_Y.T)
    return phi_X.dot(phi_Y.T)

def sum_spectrum_kernel(X, Y=None, l = 3, j_X = 0, j_Y = 0, d = None):
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
    if Y is None:
        Y = X
    if d is None:
        d = len(X[0]) 

    # sequence cannot pass the check 
    # X, Y = check_pairwise_arrays(X, Y)
    phi_X, phi_Y = Phi(X, Y, l, j_X, j_Y, d)
    return phi_X.dot(phi_Y.T)

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





