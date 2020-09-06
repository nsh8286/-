import numpy as np
a = [1,2,3,4]
b = [5,6,7,8]
a,b = np.array(a), np.array(b)
print(b-a)
print(type(b-a))