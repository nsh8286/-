import numpy as np
#import tensorflow as tf
'''
a = tf.one_hot(np.array([0,1,2,3,2,3]),4)
print (a)
predict = tf.constant(np.array([[1,1,1,99],[99,1,1,1],[2,1,99,1],[1,99,1,1],[1,1,1,99],[1,1,1,99]]))
print (predict)
print(predict*a.astype('float32'))
'''

one_hot = np.array([[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
print (one_hot)
predict = np.array([[1,1,1,99],[99,1,1,1],[2,1,99,1],[1,99,1,1],[1,1,1,99],[1,1,1,99]])
print (predict)
print(predict*one_hot)