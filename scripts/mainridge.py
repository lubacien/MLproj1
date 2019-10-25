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

#tX=kill_correlation(tX,0.98)

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
    #we save the means and standard deviations to normalise the test
    means.append(meantrain)
    devs.append(stdtrain)


weights=[]
lams=[1e-3,1e-4,1e-5,1e-6]
degs=[10,11]
losses=[]
for lam in lams:
    for deg in degs:
        print("lambda={l}, degree={d}".format(l=lam, d=deg))
        testlosses = []
        for data_set,y_set in zip(data_sets, y_sets):
            testloss, trainloss, w = cross_validation_ridge(y_set,data_set,lam,deg,0.8)
            print(testloss, trainloss)
            testlosses.append(testloss)
        losses.append([lam, deg, testlosses[0], testlosses[1], testlosses[2]])
np.asarray(losses).reshape(-1,3)



        #weights.append(w)


'''
DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
y_preds = predict_merge(tX_test,weights,means,devs)
OUTPUT_PATH = '../data/submission_splitt.csv'
create_csv_submission(ids_test, y_preds, OUTPUT_PATH)
'''
