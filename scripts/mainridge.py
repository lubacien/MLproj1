import numpy as np
from implementations import *
from cross_validation import *
from data_preprocessing import *
from proj1_helpers import *
import matplotlib.pyplot as plt
import math

print('loading data'+"\n")
DATA_TEST_PATH = '../data/train.csv'
y,tX,ids = load_csv_data(DATA_TEST_PATH)
print('data loaded')

#tX=kill_correlation(tX,0.98)

jet_set = jet(tX)

data_sets = []
y_sets = []

for i in range(len(jet_set)):
    preprocessed_tX = preprocess_data(tX[jet_set[i]])
    data_sets.append(preprocessed_tX)
    
    y_sets.append(y[jet_set[i]])

weights=[]
lams=np.logspace(-6,-2, 5)

degs = np.arange(9,13)
losses=[]

values1 = {}
values2 = {}
values3 = {}
vals = [values1,values2,values3]
for lam in lams:
    for deg in degs:
        print("lambda:{l}, degree:{d}".format(l=lam, d=deg))
        testlosses = []
        for data_set,y_set,val in zip(data_sets, y_sets,vals):
            accuracy, testloss,trainloss, w = cross_validation_ridge(y_set,data_set,lam,deg)
            print("testloss={tl}, trainloss={tr}".format(tl =testloss, tr = trainloss))
            val[accuracy] = [lam, deg]


for val in vals:
    tche = max(val)
    print(tche)
    print(val[tche])
        #weights.append(w)


'''
mse_tr = []
mse_te = []
for lam in lams:
    accuracy, testloss,trainloss, w = cross_validation_ridge(y_sets[2],data_sets[2],lam,deg,0.8)
    print("lambda={l}, testloss={tl}, trainloss={tr}".format(l=lam, tl =testloss, tr = trainloss))
    mse_tr.append(math.sqrt(trainloss))
    mse_te.append(math.sqrt(testloss))
    '''
        
'''
    """visualization the curves of mse_tr and mse_te."""
plt.semilogx(lams, mse_tr, marker=".", color='b', label='train error')
plt.semilogx(lams, mse_te, marker=".", color='r', label='test error')
plt.xlabel("lambda")
plt.ylabel("rmse")
plt.title("cross validation")
plt.legend(loc=2)
plt.grid(True)
plt.savefig("cross_validation")
'''

'''
DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
y_preds = predict_merge(tX_test,weights,means,devs)
OUTPUT_PATH = '../data/submission_splitt.csv'
create_csv_submission(ids_test, y_preds, OUTPUT_PATH)
'''
