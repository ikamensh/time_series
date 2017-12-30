import pickle
import numpy as np

from arch import arch_model

pickle_in = open("cooked/data.pickle","rb")

x_train, y_train = pickle.load(pickle_in)
x_train /= 1000
y_train /= 4000

y_train = np.swapaxes(y_train, 0, 1)

def predict_s(series, week):
    method = arch_model(y_train[series,:week])
    model = method.fit()
    a = model.forecast()
    return a.mean.iloc[-1].values


def garch_predict(n):
    forecast = np.zeros((250,))
    for i in range(y_train.shape[0]):
        forecast[i] = predict_s(i, n)
    return forecast



