from costs import compute_mse
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
    indices = random.sample(range(len(x)), int(ratio * len(x)))
    xtrain = x[indices]
    xtest = np.delete(x, indices,axis=0)
    ytrain = y[indices]
    ytest = np.delete(y, indices)
    return xtrain, ytrain, xtest, ytest


def cross_validation_for_leastsquares(y,tx,ratio):
    #we split the data for crossvalidation:
    ratio=0.8 #ratio of data used for training
    for i in range(int(1/(1-ratio))):

        xtrain, ytrain, xtest, ytest = split_data(tX, y, ratio, seed=1)

        weights_ = []
        trainlosses = []
        testlosses = []

        w, loss = least_squares(y_set,data_set)
        weights_.append(w)
        trainlosses.append(loss)
        testlosses.append(compute_loss(y_set_test,data_set_test,w))


    print("test error =",np.mean(testlosses))
    print("train error =", np.mean(trainlosses))

    #print('weights created: splitting and merging data' + "\n")
    '''
    DATA_TEST_PATH = '../data/test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    y_preds = predict_merge(tX_test,weights_)
    OUTPUT_PATH = '../data/submission_splitt.csv'
    create_csv_submission(ids_test, y_preds, OUTPUT_PATH)
    '''
    return np.mean(testlosses), np.mean(trainlosses)

def cross_validation_for_ridgereg(tX, y, lambda_, degree):

    '''print('loading data'+"\n")
    DATA_TEST_PATH = '../data/train.csv'
    y,tX,ids = load_csv_data(DATA_TEST_PATH)
    print('data loaded' + "\n")'''


    #we split the data for crossvalidation:
    ratio=0.8 #ratio of data used for training
    lambda_=0
    degrees=[3,3,3]
    for i in range(int(1/(1-ratio))):

        xtrain, ytrain, xtest, ytest = split_data(tX, y, ratio)

        jet_set = jet(xtrain)
        inds = create_inds(jet_set, False)
        data_sets = jet_split(xtrain,inds)
        y_sets = split_y(ytrain,inds)

        jet_set_test = jet(xtest)
        inds_test = create_inds(jet_set_test, False)
        data_sets_test = jet_split(xtest, inds_test)
        y_sets_test = split_y(ytest, inds_test)

        weights_ = []
        trainlosses = []
        testlosses = []
        for data_set, y_set,data_set_test,y_set_test,degree in zip(data_sets, y_sets,data_sets_test,y_sets_test,degrees):
            data_set=build_poly(data_set,degree)
            data_set_test=build_poly(data_set_test,degree)

            w, loss = ridge_regression(y_set,data_set,lambda_)

            weights_.append(w)
            trainlosses.append(loss)
            testlosses.append(compute_loss(y_set_test,data_set_test,w))


    print("test error =",np.mean(testlosses))
    print("train error =", np.mean(trainlosses))

    #print('weights created: splitting and merging data' + "\n")

   #DATA_TEST_PATH = '../data/test.csv'
    '''_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    y_preds = predict_merge(tX_test,weights_,poly=True, degrees=degrees)
    OUTPUT_PATH = '../data/submission_splitt.csv'
    create_csv_submission(ids_test, 1y_preds, OUTPUT_PATH)
    return 0
    '''
    
def cross_validation_ridge(y, tX, lambda_, degree, ratio):
    
    for i in range(int(1/(1-ratio))):

        xtrain, ytrain, xtest, ytest = split_data(tX, y, ratio)

        weights_ = []
        trainlosses = []
        testlosses = []

        data_set=build_poly(xtrain,degree)
        data_set_test=build_poly(xtest,degree)

        w, loss = ridge_regression(ytrain,data_set,lambda_)

        weights_.append(w)
        trainlosses.append(loss)
        testlosses.append(compute_loss(ytest,data_set_test,w))

    return np.mean(testlosses), np.mean(trainlosses)

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
        testlosses.append(compute_loss(ytest,data_set_test,w))

    return np.mean(testlosses), np.mean(trainlosses)
    