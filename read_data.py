
import pandas as pd
import os
from all_saturdays import all_saturdays

directory = 'data'

series = {}
for filename in os.listdir(directory):
    print('#'+filename[6:-4])


    fullname = directory + '/' + filename
    series[int(filename[6:-4])] = pd.read_csv(fullname, sep=';')


for val in series.values():
    print(val.shape)


str_saturdays = []
for j in all_saturdays(2017):
    str_saturdays.append(str(j))

# iterate over rows of each data frame, split it upon finding saturday, skip the rest of saturday.
# y = val(-1) - val(0) for the next episode.


