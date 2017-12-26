import pandas as pd
import os
from all_sundays import all_sundays
import keras

directory = 'data'
series = {}
for filename in os.listdir(directory):
    # print('#'+filename[6:-4])
    fullname = directory + '/' + filename
    series[int(filename[6:-4])] = pd.read_csv(fullname, sep=';')

def format_date( date ):
    return "{:02d}/{:02d}/{}".format(date.day, date.month, date.year)

def split_into_weeks(ser):
    sundays = all_sundays(2017, 2)
    sun = sundays.__next__()

    start = 0
    for index, row in ser.iterrows():

        if not format_date(sun) in row.Date:
            continue

        yield (ser[start:index])
        sun = sundays.__next__()
        start = index+1

def total_return(df):
    return sum(df.Our_L)

def get_x(wk):
    temp = wk.drop('Date', axis=1)
    result = temp.values
    result = result.reshape((-1,))
    return result


def omit_date(wks):
    w = list(map(get_x, wks))
    w = keras.preprocessing.sequence.pad_sequences(w, maxlen=120)
    return w

def extract_profits(wks):
    return list(map(sum, wks))

import pickle
pickle_in = open("formated.pickle","rb")
options, op_weeks, profits = pickle.load(pickle_in)

# options = [series[i] for i in range(250)]
# op_weeks = [omit_date(split_into_weeks(opt)) for opt in options ]
# profits = [extract_profits(wks) for wks in op_weeks]
# wuks = []
# for wks in op_weeks:
#     wuks.append([ wk.reshape((-1,120)) for wk in wks])
#
#
# pickle_out = open("formated.pickle", "wb")
# pickle.dump([options, wuks, profits], pickle_out)
# pickle_out.close()

import numpy as np
def train_data(weeks, profits):
    x_train = np.zeros(shape=(44,1,120), dtype=np.float32)
    y_train = np.zeros(shape=(44,1), dtype=np.float32)
    for i in range(0,len(weeks)-1-4):
        x_train[i] = weeks[i]
        y_train[i] = profits[i+1]
    return x_train, y_train

def merge_data(all_weeks, all_profits):
    xs = []
    ys = []
    for i in range(250):
        weeks = all_weeks[i]
        profits = all_profits[i]
        x, y = train_data(weeks, profits)
        xs.append(x)
        ys.append(y)

    return np.concatenate(xs), np.concatenate(ys)

x_train, y_train = merge_data(op_weeks, profits)

pickle_out = open("data.pickle", "wb")
pickle.dump([x_train, y_train], pickle_out)
pickle_out.close()

