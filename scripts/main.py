
import numpy as np
from implementations import *
from cross_validation import *
from data_preprocessing import *
from proj1_helpers import *
import math

print('loading training data'+"\n")
DATA_TEST_PATH = '../data/train.csv'
y,tX,ids = load_csv_data(DATA_TEST_PATH)
print('training data loaded'+"\n")

w_initial = np.zeros(30)

jet_tX = jet(tX)

means = []
devs = []
degree = [11, 12, 12]
# cleans -999 and standardizes
for i in range(len(jet_tX)):
    # preprocess every train subset
    preprocessed_tX = preprocess_data(tX[jet_tX[i]])
    acc, testloss, trainloss, weights = cross_validation_for_GD(y[jet_tX[i]], preprocessed_tX, degree[i])
    accs[i, :, deg - 1] = acc
    trainlosses[i, :, deg - 1] = trainloss

'''
DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
y_preds = predict_merge(tX_test,weights,means,devs)
OUTPUT_PATH = '../data/submission_splitt.csv'
create_csv_submission(ids_test, y_preds, OUTPUT_PATH)
'''