import numpy as np
from .Abstract_Layer import Abstract_Layer
from ...initializers import initializers


class Linear_Layer(Abstract_Layer):
    def __init__(self, input_dimension, output_dimension, initializer=None, weights_regularizer=None):
        Abstract_Layer.__init__(self, is_trainable=True, layer_name='Linear')

        self.input_dim, self.output_dim = input_dimension, output_dimension

        self.input_tensor, self.output_tensor = None, None
        self.input_gradients, self.output_gradients = None, None

        self.dW, self.dB = None, None

        self.optimizer = None
        self.weights_regularizer = weights_regularizer

        self.run_initializer(
            initializer=initializer, input_dimension=input_dimension, output_dimension=output_dimension)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = np.dot(self.weights, input_tensor) + self.biases

        return self.output_tensor

    def backward(self, input_gradients):
        self.input_gradients = input_gradients
        self.output_gradients = np.dot(self.weights.T, input_gradients)

        return self.output_gradients

    def apply_previous_gradient(self, input_gradients):
        self.dW = np.dot(input_gradients, self.input_tensor.T)
        self.dB = input_gradients.sum(axis=1, keepdims=True)

        if self.weights_regularizer:
            self.dW += self.weights_regularizer(self.weights)

        return self.dW, self.dB

    def optimize(self, weights_updates):
        self.weights += weights_updates[0]
        self.biases += weights_updates[1]

    def run_initializer(self, initializer, input_dimension, output_dimension):
        if not initializer:
            self.weights, self.biases = initializers.uniform(
                input_dimension=input_dimension, output_dimension=output_dimension)
        else:
            self.weights, self.biases = initializer(
                input_dimension=input_dimension, output_dimension=output_dimension)
