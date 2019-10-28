import numpy as np
import math
from proj1_helpers import *


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

            
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly



def jet(x):
    """
    Returns value corresponding to the 23 columns ( jet value
    of 0, 1, 2 and 3 ).
    """
    jet_set = {
        0: x[:, 22] == 0,
        1: x[:, 22] == 1,
        2: np.logical_or(x[:, 22] == 2,x[:, 22] == 3)
        }
    
    return jet_set

def kill_correlation(tx,thresh):
    correlationmat = np.corrcoef(tx,y=None,rowvar=False)
    ind=[]
    for i in range(tx.shape[1]):
        for j in range(i):
            if correlationmat[i,j]>=thresh:
                ind.append(j)

    tx= np.delete(tx,ind, axis=1)

    return tx

def predict_merge(tX_test, weights, y_pred, indices):
    
    y_pred[indices] = predict_labels(weights, tX_test)
    
    return y_pred

def preprocess_data(tX):
    tX = remove_features(tX)
    tX = replace_aberrant_values(tX)
    tX= kill_correlation(tX,0.95)
    tX,mean,std = standardize(tX)
    
    return(tX)

def remove_features(tX):
    """ Deletes columns that are entirely filled with aberrant values"""
    tX = np.delete(tX,(22), axis=1)
    tX = tX[:, ~(tX == tX[0,:]).all(0)]
    return tX


def replace_aberrant_values(tX):
    '''Replaces the aberrant value (-999) for a given feature 
     by the mean observed value of that feature.'''
    tX_repl_feat = np.copy(tX)
    means = []
    
    #compute the mean of each feature (column) without taking -999 values into account
    for j in range(tX_repl_feat.shape[1]):
        m = tX_repl_feat[:,j][tX_repl_feat[:,j] != -999].mean()
        means.append(m)
        
    #change all -999 values of a column by the mean computed previously
    for i in range(len(means)):
        mask = tX_repl_feat[:, i] == -999
        tX_repl_feat[:, i][mask] = means[i]
    
    return tX_repl_feat

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/(np.exp(t)+1)

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x,axis=0)
    x = x - mean_x

    std_x = np.std(x,axis=0)
    x = x / std_x
    return x, mean_x, std_x
