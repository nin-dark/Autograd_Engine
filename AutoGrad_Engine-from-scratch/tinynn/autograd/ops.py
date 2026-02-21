from .engine import Node

def add(a, b):
    out = Node(
        value = a.value + b.value,
        parents = (a,b)
    )
    def backward():
        a.grad += out.grad * 1
        b.grad += out.grad * 1
    out.backward_fn = backward
    return out

def mul(a, b):
    out = Node(
        value = a.value * b.value,
        parents = (a,b)
    )
    def backward():
        a.grad += out.grad * b.value
        b.grad += out.grad * a.value
    out.backward_fn = backward
    return out

def neg(a):
    out = Node(
        value = a.value * -1,
        parents= (a,)
    )
    def backward():
        a.grad += out.grad * -1
    out.backward_fn = backward
    return out

def pow(a, n):
    out = Node(
        value= a.value ** n,
        parents= (a,)
    )
    def backward():
        a.grad += out.grad * (n * (a.value**(n-1)))
    out.backward_fn= backward
    return out

def exp(a):
    import math
    out = Node(
        value= math.exp(a.value),
        parents= (a,)
    )
    def backward():
        a.grad += out.grad * out.value
    out.backward_fn = backward
    return out

def log(a):
    import math
    out = Node(
        value= math.log(a.value),
        parents= (a,)
    )
    def backward(): 
        a.grad += out.grad * (1/a.value)
    out.backward_fn= backward
    return out

def sin(a):
    import math
    out = Node(
        value= math.sin(a.value),
        parents= (a,)
    )
    def backward(): 
        a.grad += out.grad * math.cos(a.value)
    out.backward_fn = backward
    return out

def lifter(value):
    if isinstance(value, Node):
        return value
    else:
        return Node(value)
    
def __add__(self, other):
    other = lifter(other)
    return add(self, other)

Node.__add__ = __add__

def __mul__(self, other):
    other = lifter(other)
    return mul(self, other)

Node.__mul__ = __mul__

def __sub__(self,other):
    other = lifter(other)
    return add(self, -other)

Node.__sub__ = __sub__

def __radd__(self, other):
    other = lifter(other)
    return add(other, self)

Node.__radd__ = __radd__

def __rmul__(self, other):
    other = lifter(other)
    return mul(other,self)

Node.__rmul__ = __rmul__

def __rsub__(self,other):
    other = lifter(other)
    return add(other, -self)

Node.__rsub__ = __rsub__

def __pow__(self, other):
    return pow(self, other)

Node.__pow__ = __pow__

def __neg__(self):
    return neg(self)

Node.__neg__ = __neg__