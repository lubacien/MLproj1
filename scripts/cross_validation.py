import math
import numpy as np
import random as random
from costs import *
from data_preprocessing import *
from proj1_helpers import *
from implementations import *

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)

    indices = random.sample(range(len(x)), int(ratio * len(x)))
    xtrain = x[indices]
    xtest = np.delete(x, indices,axis=0)
    ytrain = y[indices]
    ytest = np.delete(y, indices)
    return xtrain, ytrain, xtest, ytest


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation_for_leastsquares(y,tX,degree):
    weights_ = []
    trainlosses = []
    testlosses = []
    acc = []

    seed = 1
    k_fold = 5

    k_indices = build_k_indices(y, k_fold, seed)

    for k in range(k_fold):
        # get k'th subgroup in test, others in train
        ytest = y[k_indices[k]]
        ytrain = np.delete(y, k_indices[k])
        xtest = tX[k_indices[k]]
        xtrain = np.delete(tX, k_indices[k], axis=0)

        # form data with polynomial degree
        xtestpol = build_poly(xtest, degree)
        xtrainpol = build_poly(xtrain, degree)

        # ridge regression:
        w, mse = least_squares(ytrain, xtrainpol)

        loss_tr = mse
        loss_te = compute_mse(ytest, xtestpol, w)
        trainlosses.append(loss_tr)
        testlosses.append(loss_te)

        acc.append(accuracy(ytest, xtestpol, w))
        weights_.append(w)

    print("test error:", np.mean(testlosses))
    print("train error:", np.mean(trainlosses))
    print("accuracy:", np.mean(acc))
    return np.mean(acc), np.mean(testlosses), np.mean(trainlosses), np.mean(weights_,axis=0)
    
def cross_validation_for_GD(y,tX,degree, stoch = False):
    weights_ = []
    trainlosses = []
    testlosses = []
    acc = []
    
    
    max_iters=100
    seed = 1
    k_fold = 5

    k_indices = build_k_indices(y, k_fold, seed)

    for k in range(k_fold):
        # get k'th subgroup in test, others in train
        ytest = y[k_indices[k]]
        ytrain = np.delete(y, k_indices[k])
        xtest = tX[k_indices[k]]
        xtrain = np.delete(tX, k_indices[k], axis=0)

        # form data with polynomial degree
        xtest = build_poly(xtest, degree)
        xtrain = build_poly(xtrain, degree)

        initial_w=np.zeros(xtrain.shape[1])
        if stoch == False:
            gamma = find_g(ytrain,xtrain, initial_w, [1e-7,1e-6], stoch = False)
            w, mse = least_squares_GD(ytrain, xtrain, initial_w, max_iters, gamma)
            
        if stoch == True:
            
            gamma =find_g(ytrain,xtrain,initial_w,[1e-6,1e-7], stoch = True)
            w, mse = least_squares_SGD(ytrain, xtrain, initial_w, 1, max_iters, gamma)

        loss_tr = mse
        loss_te = compute_mse(ytest, xtest, w)
        trainlosses.append(loss_tr)
        testlosses.append(loss_te)

        acc.append(accuracy(ytest, xtest, w))
        weights_.append(w)

    print("test error:", np.mean(testlosses))
    print("train error:", np.mean(trainlosses))
    print("accuracy:", np.mean(acc))
    return np.mean(acc), np.mean(testlosses), np.mean(trainlosses), np.mean(weights_,axis=0)
    
def cross_validation_ridge(y, tX, lambda_, degree):
    weights_ = []
    trainlosses = []
    testlosses = []
    acc = []

    seed = 1
    k_fold = 5
    
    k_indices = build_k_indices(y,k_fold,seed)

    for k in range(k_fold):
        # get k'th subgroup in test, others in train
        ytest=y[k_indices[k]]
        ytrain=np.delete(y,k_indices[k])
        xtest=tX[k_indices[k]]
        xtrain=np.delete(tX,k_indices[k], axis = 0)
        
        
        # form data with polynomial degree
        xtestpol=build_poly(xtest,degree)
        xtrainpol=build_poly(xtrain,degree)

        # ridge regression:
        w, mse = ridge_regression(ytrain,xtrainpol,lambda_)
        
        loss_tr = mse
        loss_te = compute_mse(ytest,xtestpol,w)
        trainlosses.append(loss_tr)
        testlosses.append(loss_te)
        
        acc.append(accuracy(ytest,xtestpol,w))
        weights_.append(w)
        
    print("test error:", np.mean(testlosses))
    print("train error:", np.mean(trainlosses))
    print("accuracy:", np.mean(acc))
    
    return np.mean(acc), np.mean(testlosses), np.mean(trainlosses), np.mean(weights_,axis=0)

def cross_validation_for_logistic(y, tX, degree):
    weights_ = []
    trainlosses = []
    testlosses = []
    acc = []

    max_iters = 100
    seed = 1
    k_fold = 5

    k_indices = build_k_indices(y, k_fold, seed)

    for k in range(k_fold):
        # get k'th subgroup in test, others in train
        ytest = y[k_indices[k]]
        ytrain = np.delete(y, k_indices[k])
        xtest = tX[k_indices[k]]
        xtrain = np.delete(tX, k_indices[k], axis=0)

        # form data with polynomial degree
        xtest = build_poly(xtest, degree)
        xtrain = build_poly(xtrain, degree)

        initial_w = np.zeros(xtrain.shape[1])
        gamma = find_g(ytrain, xtrain, initial_w, [1e-7, 1e-6])
        w, mse = logistic_regression(ytrain, xtrain, initial_w, max_iters, gamma)

        loss_tr = mse
        loss_te = compute_mse(ytest, xtest, w)
        trainlosses.append(loss_tr)
        testlosses.append(loss_te)

        acc.append(accuracy(ytest, xtest, w))
        weights_.append(w)

    print("test error:", np.mean(testlosses))
    print("train error:", np.mean(trainlosses))
    print("accuracy:", np.mean(acc))
    return np.mean(acc), np.mean(testlosses), np.mean(trainlosses), np.mean(weights_,axis=0)

def cross_validation_for_reglogistic(y, tX,lambda_, degree):
    weights_ = []
    trainlosses = []
    testlosses = []
    acc = []

    max_iters = 100
    seed = 1
    k_fold = 5

    k_indices = build_k_indices(y, k_fold, seed)

    for k in range(k_fold):
        # get k'th subgroup in test, others in train
        ytest = y[k_indices[k]]
        ytrain = np.delete(y, k_indices[k])
        xtest = tX[k_indices[k]]
        xtrain = np.delete(tX, k_indices[k], axis=0)

        # form data with polynomial degree
        xtest = build_poly(xtest, degree)
        xtrain = build_poly(xtrain, degree)

        initial_w = np.zeros(xtrain.shape[1])
        gamma = find_g(ytrain, xtrain, initial_w, [1e-7, 1e-6])
        w, mse = reg_logistic_regression(ytrain, xtrain, lambda_, initial_w, max_iters, gamma)

        loss_tr = mse
        loss_te = compute_mse(ytest, xtest, w)
        trainlosses.append(loss_tr)
        testlosses.append(loss_te)

        acc.append(accuracy(ytest, xtest, w))
        weights_.append(w)

    '''
    print("test error:", np.mean(testlosses))
    print("train error:", np.mean(trainlosses))
    print("accuracy:", np.mean(acc))
    '''
    return np.mean(acc), np.mean(testlosses), np.mean(trainlosses), np.mean(weights_,axis=0)

def find_g(y,tX, w_ini, inter, stoch = False):
    losses = []
    gammas = []
    ran = inter[0]-inter[1]
    for g in np.linspace(inter[0],inter[1], 10):
        if stoch == False:
            weight, loss = least_squares_GD(y,tX,w_ini, 2,g)
        if stoch == True:
            weight, loss = least_squares_SGD(y,tX, w_ini,1, 2, g)
        losses.append(loss)
        gammas.append(g)

    ind = losses.index(min(losses))
    gamma = gammas[ind]

    return gamma