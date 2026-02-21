from .neuron import Neuron

class Layer:
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.neurons = [Neuron(dim_in) for _ in range(dim_out)]

    def forward(self, x):
        if len(x) != self.dim_in:
            raise ValueError(
                f"Layer expected input dimension {self.dim_in}, got {len(x)}"
            )

        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(x))

        return outputs