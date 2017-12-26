from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.activations import elu
from keras.optimizers import nadam
from keras.objectives import mean_absolute_error
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

def get_allowed_info(week_n):
    if week_n < 2:
        return None, None
    if week_n >20:
        x, y = x_train[:, week_n - 20:week_n - 1], y_train[:, week_n - 20:week_n - 1]
    else:
        x, y = x_train[:, :week_n - 1], y_train[:, :week_n - 1]
    x = x.reshape(-1,1,120)
    y = y.reshape(-1,1)
    return x, y


def train_for_epochs(model, n_epochs, week_n):
    for i in range(n_epochs):
        x, y = get_allowed_info(week_n)
        x_noise = np.random.normal(scale=0.04, size=x.shape)
        y_noise = np.random.normal(scale=0.04, size=y.shape)
        model.fit(x +x_noise, y +y_noise, epochs=1, verbose=2)

# model = create_model()
# train_for_epochs(model, 3, 4)

# train_for_epochs(33)
# model.save("models/rnn_2")