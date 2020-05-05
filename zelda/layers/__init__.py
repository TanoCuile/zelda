from .BinaryCrossEntropy_Layer import BinaryCrossEntropy_Layer
from .Linear_Layer import Linear_Layer
from .ReLU_Layer import ReLU_Layer
from .Sigmoid_Layer import Sigmoid_Layer


class Layers():
    def __init__(self):
        self.Sigmoid = Sigmoid_Layer
        self.ReLU = ReLU_Layer
        self.Linear = Linear_Layer
        self.BinaryCrossEntropy = BinaryCrossEntropy_Layer

layers = Layers()
