import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras

from train_rnn import create_model, get_predictor

pickle_in = open("cooked/data.pickle","rb")

x_train, y_train = pickle.load(pickle_in)
x_test, y_test = x_train[1:], y_train[1:]

# mdl = create_model(250)


def get_predictor(seq, n_sliding):

    shape = seq.shape
    dim0 = n_sliding-1
    temp = np.concatenate([np.zeros([dim0, shape[1], shape[2]], dtype=np.float32), seq])

    predictors = []
    for wk in range(seq.shape[0]):
        slc = temp[wk:wk+n_sliding]
        slc = np.swapaxes(slc, 0,1)
        merged = slc.reshape(-1,1,n_sliding*120)
        merged = np.swapaxes(merged, 0, 1)
        predictors.append(merged)

    return np.vstack(predictors)

# mdl.evaluate(x_test[20], y_test[])



