import numpy as np
from abc import ABC, abstractmethod

class Tensor:
    def __init__(self, data, requires_grad = False, name = None):
        self.data = np.asarray(data)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self.name = name
    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

class Parameter(Tensor):
    def __init__(self, data, requires_grad = True, name = None):
        super().__init__(data, requires_grad, name)
    
class Module(ABC):
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)
    
    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, *grads):
        raise NotImplementedError
    
    def parameters(self):
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()