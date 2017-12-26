from catboost import CatBoostRegressor
import pickle
import numpy as np


pickle_in = open("cooked/data.pickle","rb")
x_train, y_train = pickle.load(pickle_in)
x_train = x_train.reshape(-1,120)
x_train /= 3000
y_train /= 4000

model = CatBoostRegressor(loss_function='MAE')


# x_noise = np.random.normal(scale=0.1, size=x_train.shape)
# y_noise = np.random.normal(scale=0.1, size=y_train.shape)
model.fit(x_train, y_train)

model.save_model("models/catboost")

