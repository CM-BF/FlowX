class parent(object):
    def __init__(self):
        pass

class child1(parent):
    def __init__(self):
        super(child1, self).__init__()
        self.value = 1

    def eval(self, child):
        if child == 1:
            print(self.value)
        elif child == 2:
            child2.eval(self)

class child2(parent):
    def __init__(self):
        super(child2, self).__init__()
        self.value = 2

    def eval(self):
        print(self.value)

c1 = child1()
c1.eval(1)
c1.eval(2)