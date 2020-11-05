# --local variable test--

def var_scope():
    global s
    print ('scope says: ',s)
    s = 'python is easy'
    print ('scope says again: ',s)

s = 'python is not easy'
print ('global says: ',s)
var_scope()
print ('global says again: ',s)