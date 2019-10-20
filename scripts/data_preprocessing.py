import numpy as np

    
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

def jet_split(x, y):
    """ Returns the splitted data sets and their previsions according to the jet number as well as the index of the data points from each set"""
    jet_set = jet(x)
    
    # Creates 3 sets of indices each corresponding to the data points belonging to the jet number
    inds0 = [i for i, x in enumerate(jet_set[0]) if x == False]
    inds1 = [i for i, x in enumerate(jet_set[1]) if x == False]
    inds2 = [i for i, x in enumerate(jet_set[2]) if x == False]
    
    tX_jet0 = np.delete(x, inds0, axis = 0)
    y_jet0 = np.delete(y, inds0, axis = 0)
    
    tX_jet1 = np.delete(x, inds1, axis = 0)
    y_jet1 = np.delete(y, inds1, axis = 0)
    
    tX_jet2 = np.delete(x, inds2, axis = 0)
    y_jet2 = np.delete(y, inds2, axis = 0)
    
    return [trim_data(tX_jet0), trim_data(tX_jet1), trim_data(tX_jet2)], [ y_jet0, y_jet1, y_jet2], [inds0, inds1, inds2]



