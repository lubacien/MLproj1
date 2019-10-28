import numpy as np
from costs import *
from gradient import *
from data_preprocessing import *

"""Implementations"""


def least_squares_GD(y, tx, initial_w, max_iters, gamma, stoch = False):

    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # computes gradient and loss

        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        

        #updates w

        w = w - gamma * grad

        # store w and loss

        ws.append(w)
        losses.append(loss)

    return w, loss


def least_squares_SGD(y, tx, initial_w, batch_size,  max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    num_batches = int(y.shape[0] / batch_size)
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    n = -1
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        n = n + 1
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        loss = compute_loss(minibatch_y, minibatch_tx, w)
        w = w - gamma * (grad / batch_size)
        # store w and loss
        
        ws.append(w)
        losses.append(loss)
        '''print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n, ti=max_iters, l=loss, w0=w[0], w1=w[1]))'''

    return w, loss


def least_squares(y, tx):
    w= np.linalg.inv(np.transpose(tx).dot(tx)).dot(np.transpose(tx)).dot(y)
    mse=compute_loss(y,tx,w)
    return w,mse



def ridge_regression(y, tx, lambda_):

    penalty = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])

    w = np.linalg.solve(tx.T.dot(tx) + penalty, tx.T.dot(y))
    
    loss = compute_loss(y, tx, w) #+lambda_*np.linalg.norm(w)**2

    return w,loss



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent"""
    
        # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    y = (y + 1) / 2  # [-1, 1] -> [0, 1]
    for n_iter in range(max_iters):
        # computes gradient and loss

        grad = compute_gradient_log(y, tx, w)
        loss = compute_loss_log(y, tx, w)

        #updates w

        w = w - gamma * grad
        # store w and loss

        ws.append(w)
        losses.append(loss)
        #print("logistic regression: Gradient Descent({bi}/{ti}): loss={l}".format(
         #   bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]), end="\r")
    
    return w, loss



def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
   
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    y = (y + 1) / 2  # [-1, 1] -> [0, 1]
    
    for n_iter in range(max_iters):
        # computes gradient and loss

        grad = compute_gradient_log(y, tx, w)+2*lambda_*np.linalg.norm(w)
        loss = compute_loss_log(y, tx, w)+ lambda_*(np.linalg.norm(w)**2)

        #updates w

        w = w - gamma * grad
        # store w and loss

        ws.append(w)
        losses.append(loss)
        #print("regularised logistic regression: Gradient Descent({bi}/{ti}): loss={l}".format(
         #   bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]), end="\r")
    return w, loss

