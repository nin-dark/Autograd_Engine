from autograd import Node
import random

class neuron:
    def __init__(self, dim=1):
        self.w = []
        self.b = Node(0)
        self.dim = dim
        for i in range (0, self.dim):
            w = Node(random.uniform(-0.1, 0.1))
            self.w.append(w)
            i += 1
    def pred(self, x):
        if len(x) != len(self.w):
            raise ValueError("Input dimension does not match neuron weight dimension")
        else:
            wx = Node(0)
            for i in range(0, self.dim):
                wx += x[i]*self.w[i]
            y = wx + self.b
            return y