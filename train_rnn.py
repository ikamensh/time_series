from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.activations import elu, sigmoid
from keras.optimizers import adam
from keras.objectives import mean_absolute_error, mean_squared_error
from keras.regularizers import l1_l2
import keras
import pickle
import numpy as np


def create_model(n_wks):
    model = Sequential()
    model.add(LSTM(128, activation=sigmoid, input_shape=(250,n_wks*120), dropout=0.3, kernel_regularizer=l1_l2(1e-4,1e-3)))
    model.add(Dense(250, kernel_regularizer=l1_l2(1e-4,1e-3)))
    model.compile(optimizer=adam(), loss=mean_squared_error)
    return model


