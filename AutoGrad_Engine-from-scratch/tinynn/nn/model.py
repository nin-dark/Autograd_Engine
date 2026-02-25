from .layer import layer
from .activations import relu

class TinyNN:
    def __init__(self):
        self.hidden = layer(2, 3)
        self.output = layer(3, 1)

    def forward(self, x):
        h = self.hidden.forward(x)
        a = [relu(node) for node in h]
        y = self.output.forward(a)[0]
        return y