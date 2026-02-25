from autograd import Node

def relu(node):
    out = Node(
        value = max(0, node.value),
        parents = (node,),
    )
    def backward():
        if node.value > 0:
            node.grad += out.grad
        else:
            node.grad += 0
    out.backward_fn = backward
    return out