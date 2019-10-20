from costs import compute_mse
import math

def test_model(y_test, y_train, X_train, X_test, weight):
    """"returns the rmse for the test and the train data"""

    rmse_tr = math.sqrt(2*compute_mse(y_train, X_train, weight))
    rmse_te = math.sqrt(2*compute_mse(y_test, X_test, weight))

    print("Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(tr=rmse_tr,te=rmse_te))


