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

# Define the parameters of ridge regression and degrees
lambdas = [0.001, 0.0001, 0.0001]
degrees_ = [10, 11, 11]

weights=[]

#cleans -999 and standardizes
for i in range(len(data_sets)):
    
    data_sets[i] = replace_aberrant_values(data_sets[i])

    data_sets[i],mean,std = standardize(data_sets[i])

    data_sets[i] = build_poly(data_sets[i], degrees_[i])

    wi, lossi = ridge_regression(y_sets[i], data_sets[i], lambdas[i])
    weights.append(wi)
    print(lossi)

print(len(weights))

DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
y_preds = predict_merge(tX_test,weights,mean,std, poly = True, degrees = degrees_)
OUTPUT_PATH = '../data/submission_ridge.csv'
create_csv_submission(ids_test, y_preds, OUTPUT_PATH)
