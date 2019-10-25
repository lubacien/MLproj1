import numpy as np
from implementations import *
from cross_validation import *
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

means=[]
devs=[]
#cleans -999 and standardizes
for i in range(len(data_sets)):
    data_sets[i] = replace_aberrant_values(data_sets[i])
    data_sets[i],meantrain,stdtrain = standardize(data_sets[i])
    means.append(meantrain)
    devs.append(stdtrain)


weights=[]
test_error, train_error, w1 = cross_validation_for_leastsquares(y_sets[0], data_sets[0],0.8)
weights.append(w1)
test_error, train_error, w2 = cross_validation_for_leastsquares(y_sets[1], data_sets[1],0.8)
weights.append(w2)
test_error, train_error, w3 = cross_validation_for_leastsquares(y_sets[2], data_sets[2],0.8)
weights.append(w3)

DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
y_preds = predict_merge(tX_test,weights,means,devs)
OUTPUT_PATH = '../data/submission_splitt.csv'
create_csv_submission(ids_test, y_preds, OUTPUT_PATH)
