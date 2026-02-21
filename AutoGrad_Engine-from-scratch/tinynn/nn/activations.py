from autograd import Node

def relu(node):
    out = Node(
        max(0, node.value),
        parents=(node,)
    )

    def _backward():
        if node.value > 0:
            node.grad += out.grad

    out._backward = _backward
    return out