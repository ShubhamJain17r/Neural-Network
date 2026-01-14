import numpy as np
from abc import ABC, abstractmethod

class Tensor:
    def __init__(self, data, requires_grad = False, name = None):
        pass
    def zero_grad(self):
        pass

class Parameter(Tensor):
    def __init__(self, data, requires_grad = True, name = None):
        super().__init__(data, requires_grad, name)
        pass
    
class Module(ABC):
    def __init__(self):
        pass

    def __setattr__(self, name, value):
        pass
    
    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, *grads):
        raise NotImplementedError
    
    def parameters(self):
        pass

    def zero_grad(self):
        pass