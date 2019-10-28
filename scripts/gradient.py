import numpy as np
from data_preprocessing import sigmoid

def compute_gradient_log(y, tx, w):
    return np.transpose(tx) @ (sigmoid(tx@w)-y)

def compute_gradient(y, tx, w):
    return (-1/(tx.shape[0]))*(np.transpose(tx).dot(y-tx.dot(w)))

def compute_stoch_gradient(y, tx, w):

    return (-1/(tx.shape[0]))*(np.transpose(tx).dot(y-tx.dot(w)))

