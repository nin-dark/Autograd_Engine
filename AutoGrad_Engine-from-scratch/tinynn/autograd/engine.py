class Node:
    def __init__(self, value, parents=(), backward_fn=None):
        self.value = value
        self.grad = 0
        self.parents = parents
        self.backward_fn = backward_fn

def post_dfs(node, order, visited):
    if node in visited:
        return
    visited.add(node)
    for parent in node.parents:
        post_dfs(parent, order, visited)
    order.append(node)

def backward(out_node):
    visited, order = set(), []
    post_dfs(out_node, order, visited)
    out_node.grad = 1
    for node in reversed(order):
        if node.backward_fn:
            node.backward_fn()

def zero_grad(output):
    visited, order = set(), []
    post_dfs(output, order, visited)
    for node in order:
        node.grad = 0