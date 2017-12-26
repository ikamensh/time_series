import pickle
import numpy as np

import keras
model = keras.models.load_model("models/rnn_2")

pickle_in = open("cooked/formated.pickle","rb")
options, op_weeks, profits = pickle.load(pickle_in)
actual_profits = np.vstack(profits)

def predict(model):
    predicted_profits = [] #predictions for the whole year per option
    for weeks in op_weeks:
        a = np.vstack(weeks)
        a = a.reshape((45,1,120))
        a = a[:-1]
        a = np.divide(a,3000)
        b = model.predict(a)
        b = np.multiply(b,4000)
        predicted_profits.append(b)

    pre_proc = [p.reshape((1,44)) for p in predicted_profits]
    pred_profits = np.vstack(pre_proc)
    pred_profits = np.concatenate([np.zeros(shape=(250, 1)), pred_profits], axis=1)
    return pred_profits

pred_profits = predict(model)

def get_indexes_of_max(array, n):
    sorted_indexes_of_best_predictions = array.argsort(axis=0)
    return sorted_indexes_of_best_predictions[-n:][::-1]

def get_elements_based_on_index(array, rows_index, n):
    columns = np.arange(0, array.shape[1], dtype=np.int64)
    columns = columns.reshape((1, columns.shape[0]))
    columns = np.repeat(columns, n, axis=0)

    return array[rows_index, columns]


n_options = 5
n_last_weeks = 8
rows = get_indexes_of_max(pred_profits[:,-n_last_weeks:], n_options)
chosen_ones = get_elements_based_on_index(actual_profits[:,-n_last_weeks:], rows, n_options)

# To see which options were chosen by the model, run:
# print(chosen_ones)

our_profit = 0
for i in range(n_options):
    our_profit += sum(chosen_ones[i])
our_profit /= n_options
print("our model allows for the profit of: " + str(our_profit)
      + " (only last {} weeks)".format(n_last_weeks))


# choosing the best options possible for comparison
n_options = 1
rows = get_indexes_of_max(actual_profits[:,-n_last_weeks:], n_options)
chosen_ones = get_elements_based_on_index(actual_profits[:,-n_last_weeks:], rows, n_options)

our_profit = 0
for i in range(n_options):
    our_profit += sum(chosen_ones[i])
our_profit /= n_options
print("maximum possible profit is: " + str(our_profit)
      + " (only last {} weeks)".format(n_last_weeks))












