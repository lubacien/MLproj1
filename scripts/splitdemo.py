import numpy as np
from implementations import *
from cross_validation import *
from data_preprocessing import *
from proj1_helpers import *
import math

print('loading data'+"\n")
DATA_TEST_PATH = '../data/train.csv'
y,tX,ids = load_csv_data(DATA_TEST_PATH)
print('data loaded' + "\n")

jet_set = jet(tX)
inds = create_inds(jet_set, False)

data_sets = jet_split(tX,inds)
y_sets = split_y(y,inds)

weights_ = []
losses = []
for data_set, y_set in zip(data_sets, y_sets):
    w, loss = least_squares(y_set,data_set)
    weights_.append(w)
    losses.append(loss)

print('weights created: splitting and merging data' + "\n")
print(losses)


DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
y_preds = predict_merge(tX_test,weights_)
OUTPUT_PATH = '../data/submission_splitt.csv'
create_csv_submission(ids_test, y_preds, OUTPUT_PATH)