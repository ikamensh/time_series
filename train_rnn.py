from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.activations import elu
from keras.optimizers import nadam
from keras.objectives import mean_absolute_error
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
pickle_in = open("cooked/data.pickle","rb")
x_train, y_train = pickle.load(pickle_in)
x_train /= 3000
y_train /= 4000

# 8 weeks as validation period
x_train, x_val, y_train, y_val = x_train[:10000], x_train[10000:], y_train[:10000], y_train[10000:]

model = Sequential()
model.add(LSTM(32, activation=elu, input_shape=(1,120), return_sequences=True ))
model.add(LSTM(16, activation=elu ))
model.add(Dense(1))
model.compile(optimizer=nadam(), loss=mean_absolute_error)

def train_for_epochs(n):
    for i in range(n):
        x_noise = np.random.normal(scale=0.05, size=x_train.shape)
        y_noise = np.random.normal(scale=0.05, size=y_train.shape)
        model.fit(x_train + x_noise, y_train + y_noise, epochs=3, verbose=2)
        print("validation loss: "+ str(model.evaluate(x_val, y_val, verbose=0)))


train_for_epochs(33)
model.save("models/rnn_2")