import numpy as np
from proj1_helpers import *
    
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

def predict_merge(tX_test, weights_):
    """ Takes as input the test dataset, splits it according to the jet number, predicts the y for each data set according to the corresponding model, and remerges the predicted data according to the test dataset"""
    
    test_jets =(jet(tX_test)) 
    inds_false = create_inds(test_jets, False) #creates the indexes to remove from the data set to form sets
                                               #according to jet number
    inds_true = create_inds(test_jets, True)   #creates the indexes of data points for each jet number, for 
                                                #reconstruction purposes
    test_sets = jet_split(tX_test, inds_false) #splits the test set into 3 different sets according to the 
                                                #data points jet number
    trues = np.concatenate(inds_true).ravel()   #indexes of each splitted data point
    
    y_preds = []
    inds_ = []
    #creates prediction according to each model and its corresponding data set
    for test_set, weight, ind_true in zip(test_sets, weights_, inds_true): 
        y_pred = predict_labels(weight, test_set)
        y_preds.extend(y_pred)
        inds_.extend(ind_true)
        
    y_preds = np.array(y_preds)
    trues, y_preds = zip(*sorted(zip(trues,y_preds))) #sorts the prediction back to the correct order
    
    return y_preds