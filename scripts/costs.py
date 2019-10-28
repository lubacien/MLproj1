from proj1_helpers import predict_labels
from implementations import *
import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss (MSE)"""
    e = y - tx.dot(w)
    return (1 / (2 * tx.shape[0])) * np.transpose(e).dot(e)

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def accuracy(ytest,xtest,w):
    preds = predict_labels(w,xtest)
    acc=np.where(ytest==preds)[0].shape[0]/ytest.shape[0]
    return acc

def compute_loss_log(y, tx, w):
    """compute the cost by negative log likelihood."""
        
    return np.log(1+np.exp(tx.dot(w))).sum()-y.T.dot(tx).dot(w).sum()

