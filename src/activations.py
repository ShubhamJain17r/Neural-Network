import numpy as np
from .layers import Layer

class Activation(Layer):
    def __init__(self):
        super().__init__()
        pass

class ReLU(Activation):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, Z):
        pass
    def backward(self, dA):
        pass

class Softmax(Activation):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, Z):
        pass
    def backward(self, dA):
        pass

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, Z):
        pass
    def backward(self, dA):
        pass

class Tanh(Activation):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, Z):
        pass
    def backward(self, dA):
        pass