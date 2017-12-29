from keras.layers import LSTM, Dense, Dropout, Reshape
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

def create_model_y(n_wks):
    model = Sequential()
    model.add(Reshape((1,n_wks), input_shape=(n_wks,)))
    model.add(LSTM(32, activation=sigmoid, dropout=0.5, kernel_regularizer=l1_l2(3e-4,3e-3), return_sequences=True))
    model.add(LSTM(16, activation=sigmoid, dropout=0.5, kernel_regularizer=l1_l2(3e-4,3e-3)))
    model.add(Dense(10, activation=elu))
    model.add(Dense(1))
    model.compile(optimizer=adam(), loss=mean_absolute_error)
    return model

def create_model_dense(n_wks):
    model = Sequential()
    model.add(Dense(64, activation=elu, input_shape=(n_wks,)))
    model.add(Dense(10, activation=elu))
    model.add(Dense(1))
    model.compile(optimizer=adam(), loss=mean_absolute_error)
    return model


