from math import exp

class Value:
    
    def __init__(self, data, _children = (), _operation = "", label = ""):
        self.data = data
        self._prev = set(_children)
        self._operation = _operation # What operation created this value
        self.label = label
        self.gradient = 0 # Assume at instantiation that all Value does not affect the output
        self._backward = lambda: None # The backpropagation function

    def __repr__(self):
        return f"Value(data = {self.data})"
    
    def __add__(self, other): # self + other
        other = other if isinstance(other, Value) else Value(other) # Allows for expressions like a + 1
        out = Value(self.data + other.data, _children = (self, other), _operation = "+")

        def _backward(): # Chain rule
            # Note: Additive to account for multivariable case of the chain rule where if a variable is used more than once, the previous gradients will be overwritten.
            self.gradient += 1.0 * out.gradient
            other.gradient += 1.0 * out.gradient

        out._backward = _backward

        return out
    
    def __radd__(self, other): # Allows for other + self
        return self + other
    
    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __mul__(self, other): # self * other
        other = other if isinstance(other, Value) else Value(other) # Allows for expressions like a * 2
        out = Value(self.data * other.data, _children = (self, other), _operation = "*")
        def _backward(): # Chain rule
            self.gradient += other.data * out.gradient
            other.gradient += self.data * out.gradient
        out._backward = _backward

        return out
    
    def __rmul__(self, other): # Accounts for 2 * a not working but a * 2 working (2.__mul__(a)) = Wrong
        return self * other # Will call a.__mul__(2) (a * 2)
    
    def __pow__(self, other): # Raises to the power of ..
        assert isinstance(other, (int, float)) # Force other to be an int or float
        out = Value(data = self.data ** other, _children = (self,), _operation = f"**{other}")

        def _backward(): # Chain rule
            self.gradient += other * (self.data ** (other - 1)) * out.gradient # Power rule d/dx(x^n) = nx^(n-1)

        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        # a / b == a * (1/b) == a * (b^-1)
        return self * other ** - 1

    def tanh(self):
        x = self.data
        tan_h = (exp(2 * x) - 1) / (exp(2 * x) + 1)
        out = Value(data = tan_h, _children = (self,), _operation = "tanh")

        def _backward(): # Chain rule
            self.gradient += (1 - tan_h ** 2) * out.gradient 

        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data 
        out = Value(exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad # Derivative of e^x is e^x, which is contained in out.data

        out._backward = _backward

        return out

    def backward(self):

        # Build topologically sorted list containing all nodes so that backpropagation can be performed on all nodes
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                # Add all children
                for child in v._prev:
                    build_topo(child)
                # Add self
                topo.append(v)

        build_topo(self)

        # Perform backpropagation starting from the output node
        self.gradient = 1.0
        for node in reversed(topo):
            node._backward()
