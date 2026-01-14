import numpy as np
from .core import Module

class Loss(Module):
    def __init__(self):
        super().__init__()
        pass

class MSE(Loss):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, Y_hat, Y):
        pass
    def backward(self):
        pass

class BinaryCrossEntropy(Loss):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, Y_hat, Y):
        pass
    def backward(self):
        pass

class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, Y_hat, Y):
        pass
    def backward(self):
        pass