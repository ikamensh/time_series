import pickle
import numpy as np

from train_rnn import create_model

pickle_in = open("cooked/data.pickle","rb")

x_train, y_train = pickle.load(pickle_in)
x_train /= 1000
y_train /= 4000
x_val, y_val = x_train[1:], y_train[1:]


def get_predictor(seq, n_sliding):

    shape = seq.shape
    dim0 = n_sliding-1
    print([dim0, shape[1], shape[2]])
    print(seq.shape)
    temp = np.concatenate([np.zeros([dim0, shape[1], shape[2]], dtype=np.float32), seq])

    predictors = []
    for wk in range(seq.shape[0]):
        slc = temp[wk:wk+n_sliding]
        slc = np.swapaxes(slc, 0,1)
        merged = slc.reshape(-1,1,n_sliding*120)
        merged = np.swapaxes(merged, 0, 1)
        predictors.append(merged)

    return np.vstack(predictors)

xs = get_predictor(x_train, 2)
xs_val = get_predictor(x_val, 2)

model = create_model(2)

def predict(n):
    return model.predict(xs_val[n-1:n], verbose=0)


def eval(n):
    y_p = predict(n)
    y = y_val[n-1:n]
    loss = np.mean(np.abs(y_p - y))
    print("predicting week {} : loss =".format(n+1) + str(loss))
    model.reset_states()
    return loss

def train(week, n_times):
    for i in range(n_times):
        model.fit(xs[2:week], y_train[2:week], epochs=1, batch_size=4, verbose=0)
        model.reset_states()

# eval(21)


def find_n_iterations(n):
    num = 0
    i=0
    while i<3:
        change = 1
        loss = eval(n)
        while change > 0:

            train(n,1)
            new_loss = eval(n)
            change  = loss - new_loss
            loss = new_loss
            num += 1
            if change > 0:
                i = 0
        i += 1
    return num - 3

# find_n_iterations(5)
# find_n_iterations(10)
# find_n_iterations(15)
# print(find_n_iterations(20))




# eval(19)
# eval(18)
#
# for i in range(5):
#     train(20+i)
#
#     eval(21+i)
#     eval(20+i)
#     eval(19+i)
#     eval(18+i)




