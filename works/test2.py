num = 100
class Test2Class():
    def __init__(self):
        self.classnum = num

    def __call__(self):
        return self.classnum
    
    def __repr__(self):
        return '{}'.format(5)