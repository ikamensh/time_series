from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.activations import elu
from keras.optimizers import nadam
from keras.objectives import mean_absolute_error
from keras.regularizers import l1_l2
import keras
import pickle
import numpy as np
pickle_in = open("cooked/data.pickle","rb")
x_train, y_train = pickle.load(pickle_in)
x_train /= 3000
y_train /= 4000

def create_model():
    model = Sequential()
    model.add(LSTM(16, activation=elu, input_shape=(1,120), return_sequences=True ))
    model.add(LSTM(8, activation=elu))
    model.add(Dense(1))
    model.compile(optimizer=nadam(), loss=mean_absolute_error)
    return model



def create_model():
    model = Sequential()
    model.add(LSTM(128, activation=elu, input_shape=(250,960) , kernel_regularizer=l1_l2(1e-3,1e-2)))
    model.add(Dense(250))
    model.compile(optimizer=nadam(), loss=mean_absolute_error)
    return model

def get_allowed_info(week_n):
    if week_n < 2:
        return None, None
    if week_n >9:
        x, y = x_train[:, week_n - 9:week_n - 1], y_train[week_n - 1]
    else:
        x, y = x_train[:, :week_n - 1], y_train[:, week_n - 1]
    x = x.reshape(250,-1)
    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=960)
    x = x.reshape(1,250, -1)
    y = y.reshape(1,250,1)

    return x, y

def get_predictor(week_n):
    if week_n < 2:
        return None
    if week_n >9:
        x = x_train[:, week_n - 8:week_n]
    else:
        x, y = x_train[:, :week_n], y_train[:, week_n]
    x = x.reshape(250,-1)
    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=960)
    x = x.reshape(1,250, -1)

    return x



def extend_datasets(n_week, x_ds, y_ds):
    if n_week <3:
        return x_ds, y_ds
    x, y = get_allowed_info(n_week)
    x_ds_new = np.concatenate([x_ds, x])
    y_ds_new = np.concatenate([y_ds, y])
    return x_ds_new, y_ds_new

def train_for_epochs(model, x, y, n_epochs):
    for i in range(n_epochs):
        model.fit(x, y, epochs=1, verbose=2)

# model = create_model()
# train_for_epochs(model, 3, 4)

# train_for_epochs(33)
# model.save("models/rnn_2")