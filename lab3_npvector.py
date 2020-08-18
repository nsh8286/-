import numpy as np

vector = [1,2,3,4,5,6,7,7,7,6]
m = np.int32(7)
a = vector==m
print (type (m))
print (type (vector))
print (a)
print('----------')
vector2 = [1,2,3,4,5,6,7,7,7,6]
m2 = 7
a2 = vector2==m2
print (type (m2))
print (type (vector2))
print (a2)

print('----------')
vector3 = np.array([1,2,3,4,5,6,7,7,7,6])
m3 = 7
a3 = vector3==m3
print (type (m3))
print (type (vector3))
print (a3)

