from autograd import Node
import random

class Neuron:
    def __init__(self, dim):
        self.dim = dim
        self.w = [Node(random.uniform(-0.1, 0.1)) for _ in range(dim)]
        self.b = Node(0)

    def forward(self, x):
        if len(x) != self.dim:
            raise ValueError(
                f"Expected input dimension {self.dim}, got {len(x)}"
            )

        out = Node(0)
        for i in range(self.dim):
            out += self.w[i] * x[i]

        out += self.b
        return out