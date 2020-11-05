import numpy as np

# numpy logical calculation test

a = np.array([1,1,0,1])
nota = np.logical_not(a)
notnota = ~nota
print (nota)
print (notnota)

# what happens if True,False array * int/float array?
b = np.array([10,20,30,40])
print (nota*b)
# True works as 1, False works as 0!

# is it works well if in different shape?
a_new = np.array([[1],[1],[0],[1]])
nota_new = np.logical_not(a_new)
b_new = np.array([[10],[20],[30],[40]])
print (nota_new*b_new)
# ofcourse it works well!

# multiply with different shapes?
print (nota*b_new)
print (nota_new*b)
# interseting result!
