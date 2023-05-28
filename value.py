from math import exp

class Value:
    
    def __init__(self, data, _children = (), _operation = "", label = ""):
        self.data = data
        self._prev = set(_children)
        self._operation = _operation # What operation created this value
        self.label = label
        self.gradient = 0.0 # Assume at instantiation that all Value does not affect the output

        # The backpropagation function
        # Note: In all of these functions, the gradients are additive to account for multivariable case of the chain rule where if a variable is used more than once, the previous gradients will be overwritten.
        self._backward = lambda: None 

    def __repr__(self):
        return f"Value(data = {self.data})"
    
    def __add__(self, other): # self + other
        other = other if isinstance(other, Value) else Value(other) # Allows for expressions like a + 1
        out = Value(self.data + other.data, _children = (self, other), _operation = "+")

        def _backward(): # Chain rule

            # Example: c = a + b (Assume dL/dc is an arbitrary value -2.0)
            # dL/ da = dc/da * dL/dc
            # dL/ db = dc/db * dL/dc

            # dc / da = 1.0
            # dc / db = 1.0
            
            # Therefore: dL/ da = 1.0 * -2.0, dL/db = 1.0 * -2.0
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
            # Example: c = a * b (Assume dL/dc is an arbitrary value -2.0)
            # dL/ da = dc/da * dL/dc
            # dL/ db = dc/db * dL/dc

            # dc / da = b
            # dc / db = c
            
            # Therefore: dL/ da = b * -2.0, dL/db = c * -2.0 (Values of c and b are held within .data attribute)

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
            # The gradient of this would be dL/dx = dL/dy * dy/dx, where dL/dy is the gradient of "out" and dy/dx is the local derivative
            self.gradient += out.data * out.gradient # Derivative of e^nx is ne^nx, e^x is contained in out.data

        out._backward = _backward

        return out

    def backward(self): # Alters the gradients of all weights and biases to find a 

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
        self.gradient = 1.0 # dLoss/ dLoss = 1.0
        for node in reversed(topo):
            node._backward()
