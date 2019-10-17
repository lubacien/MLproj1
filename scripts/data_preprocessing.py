{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Useful starting lines\n",
    "%matplotlib inline\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from implementations import *\n",
    "%load_ext autoreload\n",
    "%autoreload 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from proj1_helpers import *\n",
    "DATA_TRAIN_PATH = '../data/train/train.csv' # TODO: download train data and supply path here \n",
    "y, tX, ids = load_csv_data(DATA_TRAIN_PATH)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 138.47 ,   51.655,   97.827, ...,    1.24 ,   -2.475,  113.497],\n",
       "       [ 160.937,   68.768,  103.235, ..., -999.   , -999.   ,   46.226],\n",
       "       [-999.   ,  162.172,  125.953, ..., -999.   , -999.   ,   44.251],\n",
       "       ...,\n",
       "       [ 105.457,   60.526,   75.839, ..., -999.   , -999.   ,   41.992],\n",
       "       [  94.951,   19.362,   68.812, ..., -999.   , -999.   ,    0.   ],\n",
       "       [-999.   ,   72.756,   70.831, ..., -999.   , -999.   ,    0.   ]])"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tX"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "def replace_aberrant_values(tX):\n",
    "    '''Replaces the aberrant value (-999) for a given feature \n",
    "    and  replaces it by the mean observed value of that feature.'''\n",
    "    tX_repl_feat = np.copy(tX)\n",
    "    means = []\n",
    "    \n",
    "    #compute the mean of each feature (column) without taking -999 values into account\n",
    "    for j in range(tX_repl_feat.shape[1]):\n",
    "        m = tX_repl_feat[:,j][tX_repl_feat[:,j] != -999].mean()\n",
    "        means.append(m)\n",
    "    \n",
    "    #change all -999 values of a column by the mean computed previously\n",
    "    for i in range(len(means)):\n",
    "        mask = tX_repl_feat[:, i] == -999\n",
    "        tX_repl_feat[:, i][mask] = means[i]\n",
    "    \n",
    "    return tX_repl_feat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "def delete_aberrant_features(tX, frac):\n",
    "    if (frac < 0) or (frac > 1):\n",
    "        print('Fraction is not correct.')\n",
    "        return tX\n",
    "    \n",
    "    tX_del_feat = np.copy(tX)\n",
    "    feat_to_delete = []\n",
    "    \n",
    "    for i in range(tX_del_feat.shape[1]):\n",
    "        if (np.count_nonzero(tX_del_feat[:,i] == -999)/tX_del_feat.shape[0]) > frac:\n",
    "            feat_to_delete.append(i)\n",
    "    \n",
    "    tX_del_feat = np.delete(tX_del_feat, feat_to_delete, 1)\n",
    "    \n",
    "    return tX_del_feat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def redistribute_aberrant_values(tX):"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
