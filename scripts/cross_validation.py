from costs import *
import math
import numpy as np
import random as random
from data_preprocessing import *
from proj1_helpers import *

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



def cross_validation_for_leastsquares(y,tX, ratio):
    #we split the data for crossvalidation:
    weights_ = []
    trainlosses = []
    testlosses = []
    acc=[]
    for i in range(int(1/(1-ratio))):

        xtrain, ytrain, xtest, ytest = split_data(tX, y, ratio, seed=i)

        w, loss = least_squares(ytrain,xtrain)
        weights_.append(w)
        trainlosses.append(loss)
        testlosses.append(compute_loss(ytest,xtest,w))
        acc.append(accuracy(ytest,xtest,w))


    print("test error =",np.mean(testlosses))
    print("train error =", np.mean(trainlosses))
    print("accuracy = ", np.mean(acc),np.std(acc))
    #print('weights created: splitting and merging data' + "\n")
    '''
    DATA_TEST_PATH = '../data/test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    y_preds = predict_merge(tX_test,weights_)
    OUTPUT_PATH = '../data/submission_splitt.csv'
    create_csv_submission(ids_test, y_preds, OUTPUT_PATH)
    '''

    return np.mean(testlosses), np.mean(trainlosses), np.mean(weights_,axis=0)


    
def cross_validation_ridge(y, tX, lambda_, degree, ratio):
    weights_ = []
    trainlosses = []
    testlosses = []
    acc = []
    
    
    seed = 1
    k_fold = 4
    
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
        
    '''
    for i in range(int(1/(1-ratio))):

        xtrain, ytrain, xtest, ytest = split_data(tX, y, ratio, seed = i)

        data_set=build_poly(xtrain,degree)
        data_set_test=build_poly(xtest,degree)

        w, loss = ridge_regression(ytrain,data_set,lambda_)

        weights_.append(w)
        trainlosses.append(loss)
        testlosses.append(compute_loss(ytest,data_set_test,w))
        acc.append(accuracy(ytest, data_set_test, w))
        '''

    return np.mean(acc), np.mean(testlosses),np.mean(trainlosses), np.mean(weights_,axis=0)

def cross_validation_log(y, tX, lambda_, degree, ratio, gamma):
    
    maxiter=10
    init_w = np.zeros(tX.shape[1])
    
    for i in range(int(1/(1-ratio))):

        xtrain, ytrain, xtest, ytest = split_data(tX, y, ratio)

        weights_ = []
        trainlosses = []
        testlosses = []

        data_set=build_poly(xtrain,degree)
        data_set_test=build_poly(xtest,degree)
        
        w, loss = reg_logistic_regression(ytrain,data_set,lambda_, init_w,maxiter,gamma)
        
        weights_.append(w)
        trainlosses.append(loss)
        testlosses.append(compute_mse(ytest,data_set_test,w))

    return np.mean(testlosses), np.mean(trainlosses)
    