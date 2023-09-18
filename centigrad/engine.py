import numpy as np


class Value:
    def __init__(self, data: float, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        data = self.data + other.data
        out = Value(data=data, _children=(self, other), _op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        data = self.data * other.data
        out = Value(data=data, _children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        data = self.data ** other
        out = Value(data=data, _children=(self,), _op=f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**(-1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self**-1

    def __lt__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        return self.data < other.data

    def __gt__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        return self.data > other

    def __le__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        return self.data <= other.data

    def __ge__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        return self.data >= other

    def exp(self):
        x = self.data
        data = np.exp(x)
        out = Value(data=data, _children=(self,), _op='exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        x = self.data
        assert x > 0, "cant log a negative number"
        data = np.log(x)
        out = Value(data=data, _children=(self,), _op='log')

        def _backward():
            self.grad += (1 / x) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        data = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        out = Value(data=data, _children=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - data ** 2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        x = self.data
        data = np.maximum(0, x)
        out = Value(data=data, _children=(self,), _op='relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        x = self.data
        data = 1 / (1 + np.exp(-x))
        out = Value(data=data, _children=(self,), _op='sigmoid')

        def _backward():
            self.grad += (data * (1 - data)) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
