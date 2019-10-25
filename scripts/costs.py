# -*- coding: utf-8 -*-
"""A function to compute the cost."""
from proj1_helpers import predict_labels
import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def accuracy(ytest,xtest,w):
    preds = predict_labels(w,xtest)
    acc=np.where(ytest==preds)[0].shape[0]/ytest.shape[0]
    return acc