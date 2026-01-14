import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        pass

class Adam(Optimizer):
    def __init__(self, *parameters):
        super().__init__()
        pass
    def step(self):
        pass

class SGD(Optimizer):
    def __init__(self, *parameters):
        super().__init__()
        pass
    def step(self):
        pass