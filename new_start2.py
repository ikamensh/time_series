import pickle
import numpy as np
from scipy import signal


from train_rnn import create_model_y, create_model_dense

pickle_in = open("cooked/data.pickle","rb")

predictor, profits = pickle.load(pickle_in)

predictor /= 500
predictor = np.swapaxes(predictor, 0, 1)
predictor_d1 = np.diff(predictor, 1, axis=2)
predictor_fft = np.fft.fft(predictor, 2, axis=2)

features = []

array_feats = []
array_feats.append(predictor)
# array_feats.append(predictor_d1)
# array_feats.append(predictor_fft)

for feat in array_feats:
    features.append(signal.resample(feat,3, axis= 2))
    features.append(np.max(feat, axis=2, keepdims=True))
    features.append(np.min(feat, axis=2, keepdims=True))

def time_slice3d(array, length):
    x_train = array
    x_train_exp = np.concatenate([np.zeros((250,length-1, x_train.shape[2])), x_train], axis=1)
    x_train_seq = np.zeros((250,45,length,x_train.shape[2]), dtype=np.float32)
    for i in range(45):
        x_train_seq[:,i] = x_train_exp[:,i:i+length]
    return x_train_seq

feats_np = np.concatenate(features, axis=2)
feats_np = time_slice3d(feats_np, 2).reshape(250,45,-1)


profits /= 4000
profits = np.swapaxes(profits, 0, 1)

weekly_means = np.repeat(np.mean(profits, axis=0, keepdims=True), 250, axis=0)
profits = profits - weekly_means

n_seq = 12
n_epochs = 2

def time_slice(array, length):
    x_train = array
    x_train_exp = np.concatenate([np.zeros((250,length-1)), x_train], axis=1)
    x_train_seq = np.zeros((250,45,length), dtype=np.float32)
    for i in range(45):
        x_train_seq[:,i] = x_train_exp[:,i:i+length]
    return x_train_seq


x = time_slice(profits, n_seq)
x = np.concatenate([x, feats_np], axis = 2)



model = create_model_dense(x.shape[2])

y = np.concatenate([  profits[:,1:] , np.zeros( (250, 1), dtype=np.float32) ],axis = 1)


def concat_train_data(wk, start_wk=0):
    x_t = np.concatenate([x[:, i] for i in range(start_wk, wk-1)])
    y_t = np.concatenate([y[:,i] for i in range(start_wk, wk-1)])
    return x_t, y_t

def get_eval_data(wk):
    x_e = np.concatenate([x[:, i] for i in range(wk,45)])
    y_e = np.concatenate([y[:,i] for i in range(wk,45)])
    return x_e, y_e

# x_eval, y_eval = get_eval_data(20)
# x_t, y_t = concat_train_data(20)
#
# print(model.evaluate(x_eval,y_eval, verbose=2))
# # for i in range(10):
# model.fit(x_t, y_t, epochs=10, verbose=2)
# print(model.evaluate(x_eval,y_eval, verbose=2))


def predict(wk):
    return model.predict(x[:,wk-1])

def train(wk):
    x_t, y_t = concat_train_data(wk, max(wk//2-3, 0))
    model.fit(x_t, y_t, epochs=n_epochs, verbose=2)












