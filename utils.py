import pickle
import numpy as np
import scipy.sparse as sp

def save(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)
        
def feature_weighted_classification_matrix(p, f):
    n = p.shape[0] * (p.shape[2] - 1) * f.shape[1]
    X = None
    for k in range(f.shape[1]):
        for i in range(p.shape[0]):
            for j in range(1, p.shape[2]):
                if sp.issparse(f):
                    p_sparse = sp.csr_matrix(p[i,:,j]).T
                    p_sparse.multiply(f[:,k].T)
                    
                    
                    if X is None:
                        X = p_sparse
                    else:
                        X = sp.csr_matrix( sp.hstack((X, p_sparse)) )
                else:
                    if X is None:
                        X = p[i,:,j] * f[:,k]
                    else:
                        X = np.column_stack((X, p[i,:,j] * f[:,k]))  
    return X

def feature_weighted_regression_matrix(p, f):
    n = p.shape[0] * f.shape[1]
    X = None
    for k in range(f.shape[1]):
        for i in range(p.shape[0]):
            if sp.issparse(f):
                p_sparse = sp.csr_matrix(p[i,:]).multiply(f[:,k])
                if X is None:
                    X = p_sparse
                else:
                    X = sp.csr_matrix( sp.hstack((X, p_sparse)) )
            else:
                if X is None:
                    X = p[i,:] * f[:,k]
                else:
                    X = np.column_stack((X, p[i,:] * f[:,k]))     
    return X 
    
def feature_prediction_classification_concat(p, f):
    X = f
    for i in range(p.shape[0]):
        for j in range(1, p.shape[2]):
            if sp.issparse(X):
                p_sparse = sp.csr_matrix(p[i,:,j]).T
                X = sp.csr_matrix( sp.hstack((X, p_sparse)) )
            else:
                X = np.column_stack((X, p[i,:,j]))
    return X
    
def feature_prediction_regression_concat(p, f):
    X = f
    for i in range(p.shape[0]):
        if sp.issparse(X):
            p_sparse = sp.csr_matrix(p[i,:])
            X = sp.csr_matrix( sp.hstack((X, p_sparse)) )
        else:
            X = np.column_stack((X, p[i,:]))
    return X
