
import pandas as pd
import os
from all_sundays import all_sundays
import numpy as np

directory = 'data'

series = {}
for filename in os.listdir(directory):
    # print('#'+filename[6:-4])
    fullname = directory + '/' + filename
    series[int(filename[6:-4])] = pd.read_csv(fullname, sep=';')

# for val in series.values():
#     print(val.shape)

sundays = all_sundays(2017, 2)

def format_date( date ):
    return "{:02d}/{:02d}/{}".format(date.day, date.month, date.year)

def split_into_weeks(ser):

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

def xy_train(weeks):
    ys = []
    for week in weeks:
        ys.append(total_return(week))

    ys.append(0)

    for i, w in enumerate(weeks):
        x = w.Our_L.values
        y = ys[i+1]
        yield x, y

datasets = []
weeks = list(split_into_weeks(series[0]))

# for df in series.values():


def dataset_from_weeks(wks):
    x_train_tmp = []
    x_train = []
    y_train = []

    xy_gen = xy_train(wks)
    for x, y in xy_gen:
        x_train_tmp.append(x)
        y_train.append(y)

    for x in x_train:
        x.reshape( (1,-1), inplace=True)

    for x in x_train:
        print(x.shape)

dataset_from_weeks(weeks)

# iterate over rows of each data frame, split it upon finding saturday, skip the rest of saturday.
# y = val(-1) - val(0) for the next episode.


