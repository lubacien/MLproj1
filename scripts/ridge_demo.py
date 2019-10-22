import numpy as np
from implementations import *
from cross_validation import test_model
from data_preprocessing import *
from proj1_helpers import *
import math

print('loading data'+"\n")
DATA_TEST_PATH = '../data/train.csv'
y,tX,ids = load_csv_data(DATA_TEST_PATH)
print('data loaded')

jet_set = jet(tX)
inds = create_inds(jet_set, False)

data_sets = jet_split(tX,inds)
y_sets = split_y(y,inds)

weights_ = []
losses = []
lambda_=10000000

degree = 3
for data_set, y_set in zip(data_sets, y_sets):
    tXpol = build_poly(data_set, degree)
    w, loss = ridge_regression(y_set,tXpol,lambda_)
    weights_.append(w)
    losses.append(loss)



DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
y_preds = predict_merge(tX_test,weights_, poly = True, degree = 3)
OUTPUT_PATH = '../data/submission_splitt.csv'
create_csv_submission(ids_test, y_preds, OUTPUT_PATH)
