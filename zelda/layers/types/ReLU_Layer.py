import numpy as np
from .Abstract_Layer import Abstract_Layer


class ReLU_Layer(Abstract_Layer):
    def __init__(self):
        Abstract_Layer(self, layer_name='ReLU')
        self.input_tensor, self.output_tensor = None, None
        self.input_gradients, self.output_gradients = None, None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = np.maximum(input_tensor, 0)

        return self.output_tensor

    def backward(self, input_gradients):
        self.input_gradients = input_gradients
        self.output_gradients = self.input_gradients * \
            np.where(self.output_tensor <= 0, 0, 1)

        return self.output_gradients
