from .Model import Model
from .optimizers import optimizers
from .layers import layers


class Zelda():
    def __init__(self):
        self.Model = Model
        self.optimizers = optimizers
        self.layers = layers


zelda = Zelda()
