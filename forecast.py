import pickle
import numpy as np
from new_start3 import train, predict


pickle_in = open("cooked/formated.pickle","rb")
options, op_weeks, profits = pickle.load(pickle_in)
pickle_in.close()
actual_profits = np.vstack(profits)

#TODO verdächtig!
def predict_rnn():
    pred_profits = np.zeros((250,43)) #predictions for the whole year per option

    for wk in range(3, 45):
        print("predicting week {}".format(wk))
        train(wk)
        b = predict(wk)
        b = np.multiply(b,4000)
        pred_profits[:,wk-2] = b.flatten()

    # first two weeks are not predicted at all.
    pred_profits = np.concatenate([np.zeros(shape=(250, 2)), pred_profits], axis=1)
    return pred_profits

def predict_avg(n):
    pred_profits = np.zeros((250, 45))  # predictions for the whole year per option
    for wks in range(1,45):
        if wks < n:
            pred_profits[:,wks] = np.average(actual_profits[:,:wks], axis=1)
        else:
            pred_profits[:, wks] = np.average(actual_profits[:, wks-n:wks], axis=1)
    return pred_profits



def get_indexes_of_max(array, n):
    sorted_indexes_of_best_predictions = array.argsort(axis=0)
    return sorted_indexes_of_best_predictions[-n:][::-1]

def get_elements_based_on_index(array, rows_index, n):
    columns = np.arange(0, array.shape[1], dtype=np.int64)
    columns = columns.reshape((1, columns.shape[0]))
    columns = np.repeat(columns, n, axis=0)

    return array[rows_index, columns]

def choose_n_best_ones(n, predictions, real, msg):

    rows = get_indexes_of_max(predictions, n)
    chosen_ones = get_elements_based_on_index(real, rows, n)

    our_profit = 0
    for i in range(n):
        our_profit += sum(chosen_ones[i])
    our_profit /= n
    print(msg + str(our_profit))



pred_profits_avg = predict_avg(6)
pred_profits_rnn =  predict_rnn()


def test_predictions(predictions):
    choose_n_best_ones(1 ,predictions , actual_profits, "1our model allows for the profit of: ")
    choose_n_best_ones(2 ,predictions , actual_profits, "2our model allows for the profit of: ")
    choose_n_best_ones(5 ,predictions , actual_profits, "5our model allows for the profit of: ")
    choose_n_best_ones(15 ,predictions , actual_profits, "15our model allows for the profit of: ")

# test_predictions(pred_profits_avg)
test_predictions(pred_profits_rnn)
# pred = np.concatenate([pred_profits_avg[:,:15],pred_profits_rnn[:,15:]], axis=1)
# test_predictions(pred)

# test_predictions(pred_profits_avg - pred_profits_rnn)
# test_predictions(pred_profits_avg + pred_profits_rnn)


# choosing the best options possible for comparison
choose_n_best_ones(1 ,actual_profits , actual_profits, "maximum possible profit is: ")
choose_n_best_ones(250 ,actual_profits , actual_profits, "expected profit of a random choice: ")













