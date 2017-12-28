import os

def findfiles(rootdir):
    for filename in os.listdir(rootdir):
        path = os.path.abspath(rootdir +'/'+filename)
        print('what is  ' + path +'?')
        if os.path.isfile(path):
            print('file: ' + path)
            yield path
        elif os.path.isdir(path):
            print('dir: '+path)
            yield from findfiles(path)



# g = findfiles(r'C:\Users\ikamensh\game_nidolons')

import numpy as np

def average(arr, n, axis):



    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), axis)

A = np.array([[1,2,3],[4,5,6]])
print(A)

