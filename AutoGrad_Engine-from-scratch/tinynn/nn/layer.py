from .neuron import neuron

class layer:
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.neurons = [neuron(dim_in) for _ in range(dim_out)]
    
    def forward(self, x):
        yout = []
        if len(x) != self.dim_in:
            raise ValueError(f"Layer expected input dimension {self.dim_in}, got {len(x)}")
        else:
            for neuron in self.neurons:
                yout.append(neuron.pred(x))
        return yout