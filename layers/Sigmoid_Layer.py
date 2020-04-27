import numpy as np
from .Abstract_Layer import Abstract_Layer


class Sigmoid_Layer(Abstract_Layer):
    def __init__(self):
        Abstract_Layer.__init__(self, is_trainable=True, layer_name='Sigmoid')
        self.input_tensor, self.output_tensor = None, None
        self.input_gradients, self.output_gradients = None, None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = 1 / (1 + np.exp(-input_tensor))

        return self.output_tensor

    def backward(self, input_gradients):
        self.input_gradients = input_gradients
        self.output_gradients = self.input_gradients * \
            (self.output_tensor * (1 - self.output_tensor))

        return self.output_gradients
