import numpy as np
import random as rand
from implementations import *

y, features = load_data()

features, mean_feat, std_feat = standardize(features)
y ,features = build_model_data(y,features)
rand.seed(a=2)

w_init=np.random.rand(np.shape(features)[1])

#least_squares_GD(y,features,w_init,50,0.1)

#least_squares_SGD(y, features, w_init, 100, 50, 0.1)

print(least_squares(y,features))