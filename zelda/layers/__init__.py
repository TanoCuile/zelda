from .types import BinaryCrossEntropy_Layer, Linear_Layer, ReLU_Layer, Sigmoid_Layer


class Layers():
    def __init__(self):
        self.Sigmoid = Sigmoid_Layer.Sigmoid_Layer
        self.ReLU = ReLU_Layer.ReLU_Layer
        self.Linear = Linear_Layer.Linear_Layer
        self.BinaryCrossEntropy = BinaryCrossEntropy_Layer.BinaryCrossEntropy_Layer


layers = Layers()
