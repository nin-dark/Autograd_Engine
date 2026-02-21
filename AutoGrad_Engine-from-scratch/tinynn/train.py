from autograd.engine import backward, zero_grad
from tinynn.nn import TinyNN

dataset = [
    (2, 3, 16),
    (1, 1, 5)
]

model = TinyNN()
lr = 0.01

for step in range(100):
    for x1, x2, ytrue in dataset:
        x = [x1, x2]

        # ---- ZERO GRADS ----
        for neuron in model.hidden.neurons:
            for w in neuron.w:
                zero_grad(w)
            zero_grad(neuron.b)

        for neuron in model.output.neurons:
            for w in neuron.w:
                zero_grad(w)
            zero_grad(neuron.b)

        # ---- FORWARD ----
        y = model.forward(x)
        loss = (y - ytrue) ** 2

        # ---- BACKWARD ----
        backward(loss)

        # ---- UPDATE ----
        for neuron in model.hidden.neurons:
            for w in neuron.w:
                w.value -= lr * w.grad
            neuron.b.value -= lr * neuron.b.grad

        for neuron in model.output.neurons:
            for w in neuron.w:
                w.value -= lr * w.grad
            neuron.b.value -= lr * neuron.b.grad

    print("Step:", step, "Loss:", loss.value)