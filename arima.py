import pickle
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm

pickle_in = open("cooked/data.pickle","rb")

x_train, y_train = pickle.load(pickle_in)
x_train /= 1000
y_train /= 4000

y_train = np.swapaxes(y_train, 0, 1)

# tmp_mdl = smt.ARIMA(y_train[0, :12], order=(0, 1, 1)).fit(method='mle', trend='nc')


def best_model(series, week):

    best_aic = np.inf
    best_order = None
    best_mdl = None

    pq_rng = range(5) # [0,1,2,3]
    d_rng = range(2) # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(y_train[series,:week], order=(i,d,j)).fit(method='mle', trend='nc')
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue


    return best_mdl

# p = []
#
# p.append(best_params(12, 14))
# p.append(best_params(112, 22))
# p.append(best_params(212, 38))
# p.append(best_params(152, 21))
# p.append(best_params(182, 14))


def arima_predict(n):
    forecast = np.zeros((250,))
    for i in range(y_train.shape[0]):
        tmp_mdl = best_model(i, n)
        if tmp_mdl is None:
            continue
        forecast[i] = tmp_mdl.forecast()[0]
    return forecast




