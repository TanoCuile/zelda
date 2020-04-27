import numpy as np
from .Abstract_Layer import Abstract_Layer


class BinaryCrossEntropy_Layer(Abstract_Layer):
    def __init__(self):
        Abstract_Layer(self, layer_name='Cross Entropy')
        self.P = None
        self.Y = None
        self.m = None

    def forward(self, predicted, Y):
        self.P = predicted
        self.Y = Y
        self.batch_size, self.number_of_classes = Y.shape

        loss = - (1 / self.batch_size) * (1 / self.number_of_classes) * \
            np.sum(self.Y * np.log(self.P).T +
                   (1 - self.Y) * np.log(1 - self.P).T)
        return loss

    def backward(self):
        # * self.number_of_classes)
        return ((self.P - self.Y.T) / (self.P * (1 - self.P))) / (self.batch_size)
