import numpy as np
import math
from proj1_helpers import *
from implementations import *


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


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x,axis=0)
    x = x - mean_x

    std_x = np.std(x,axis=0)
    x = x / std_x
    return x, mean_x, std_x

'''

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


def trim_data(tX):
    """ Deletes columns that are entirely filled with aberrant values"""
    tX = np.delete(tX,(22), axis=1)
    tX = tX[:, ~(tX == tX[0,:]).all(0)]
    return tX

def create_inds(jet_set, boolval):
    inds0 = [i for i, x in enumerate(jet_set[0]) if x == boolval]
    inds1 = [i for i, x in enumerate(jet_set[1]) if x == boolval]
    inds2 = [i for i, x in enumerate(jet_set[2]) if x == boolval]

    return [inds0, inds1, inds2]

def jet_split(x, inds):
    """ Returns the splitted data sets and their previsions according to the jet number as well as the index of the data points from each set"""
    
    # Creates 3 sets of indices each corresponding to the data points belonging to the jet number
    tX_jet0 = np.delete(x, inds[0], axis = 0)
    tX_jet1 = np.delete(x, inds[1], axis = 0)
    tX_jet2 = np.delete(x, inds[2], axis = 0)
    
    return [trim_data(tX_jet0), trim_data(tX_jet1), trim_data(tX_jet2)]


def split_y(y, inds):
    y_jet0 = np.delete(y, inds[0], axis = 0)
    y_jet1 = np.delete(y, inds[1], axis = 0)
    y_jet2 = np.delete(y, inds[2], axis = 0)
    
    return [y_jet0, y_jet1, y_jet2]

def predict_merge(tX_test, weights_,means,devs,poly, degrees):
    """ Takes as input the test dataset, splits it according to the jet number, predicts the y for each data set according to the corresponding model, and remerges the predicted data according to the test dataset"""
    
    test_jets =(jet(tX_test))

    inds_false = create_inds(test_jets, False) #creates the indexes to remove from the data set to form sets according to jet number
    inds_true = create_inds(test_jets, True)   #creates the indexes of data points for each jet number, for reconstruction purposes
    test_sets = jet_split(tX_test, inds_false) #splits the test set into 3 different sets according to the jet number

    #we also need to normalize the test data

    for i in range(len(test_sets)):
        test_sets[i] = replace_aberrant_values(test_sets[i])
        test_sets[i] = standardizetest(test_sets[i],means[i],devs[i])

    if poly == True:
        poly_set = []                                 
        for test_set,degree in zip(test_sets,degrees):
            poly_set.append(build_poly(test_set, degree))
        test_sets = poly_set
        print(poly_set)

    trues = np.concatenate(inds_true).ravel()   #indexes of each splitted data point
    
    y_preds = []
    inds_ = []

    #creates prediction according to each model and its corresponding data set
    for test_set, weight, ind_true in zip(test_sets, weights_, inds_true): 
        y_pred = predict_labels(weight, test_set)
        y_preds.extend(y_pred)
        inds_.extend(ind_true)
        #rmse_.append(math.sqrt(2*compute_mse(y_pred, test_set,weight)))

    y_preds = np.array(y_preds)
    trues, y_preds = zip(*sorted(zip(trues,y_preds))) #sorts the prediction back to the correct order
    
    return y_preds


def replace_aberrant_values(tX):
    #Replaces the aberrant value (-999) for a given feature 
    #and  replaces it by the mean observed value of that feature.
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

def standardize(x):

    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)

    return std_data, np.mean(x, axis=0), np.std(centered_data, axis=0)

def standardizetest(x,trainingmean,trainingdev):

    centered_data = x - trainingmean
    std_data = centered_data / trainingdev

    return std_data
'''