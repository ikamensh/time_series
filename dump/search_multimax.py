inpt = [1, 23 , -2 , 11, 19, 12]
n_max=3

array = [ (i,num) for i, num in enumerate(inpt)]
array.sort(key=lambda x: x[1], reverse=True)
indexes = [i for i, _ in array[:3]]



import numpy as np
b = np.array([[1, 3, 4], [17, 2 , 12], [3, 15, 5]])
b.argsort()


