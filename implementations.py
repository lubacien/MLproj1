import numpy as np

def load_data(sub_sample=True, add_outlier=False):

    path_dataset = "train.csv"
    features = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=range(2,32,1))

    y = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1],
        converters={1: lambda x: 0 if b"s" in x else 1})

    return y, features

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x,axis=0)
    print(mean_x.shape)
    x = x - mean_x

    std_x = np.std(x,axis=0)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(y,tx):
    """Form (y,tX) to get regression data in matrix form. (adds 1s for the w0)"""

    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), tx]
    return y, tx

"""GRADIENT DESCENT"""

def compute_gradient(y, tx, w):
    return (-1/(tx.shape[0]))*(np.transpose(tx).dot(y-tx.dot(w)))


def compute_loss(y, tx, w):
    """Calculate the loss (MSE)
    """
    e = y - tx.dot(w)
    return (1 / (2 * tx.shape[0])) * np.transpose(e).dot(e)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):

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
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss #we return only the last loss and w

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_stoch_gradient(y, tx, w):

    return (-1/(tx.shape[0]))*(np.transpose(tx).dot(y-tx.dot(w)))

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
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n, ti=max_iters, l=loss, w0=w[0], w1=w[1]))

    return w, loss

def least_squares(y, tx):
    w=np.linalg.inv(np.transpose(tx).dot(tx)).dot(np.transpose(tx)).dot(y)
    mse=compute_loss(y,tx,w)
    return mse, w

def ridge_regression(y, tx, lambda_ ):

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):

    return w, loss

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):

    return w, loss