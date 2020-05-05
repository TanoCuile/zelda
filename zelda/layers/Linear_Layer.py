import numpy as np
from .Abstract_Layer import Abstract_Layer


class Linear_Layer(Abstract_Layer):
    def __init__(self, input_dim, output_dim, initializer=None, weights_regularizer=None):
        Abstract_Layer.__init__(self, is_trainable=True, layer_name='Linear')

        self.input_dim, self.output_dim = input_dim, output_dim

        self.input_tensor, self.output_tensor = None, None
        self.input_gradients, self.output_gradients = None, None

        self.dW, self.dB = None, None

        self.optimizer = None
        self.weights_regularizer = weights_regularizer

        if not initializer:
            w_mu, w_sigma = 0., 0.01
            b_mu, b_sigma = 0.0001, 0.005
            self.weights = np.random.uniform(
                w_mu, w_sigma, (output_dim, input_dim))
            self.biases = np.random.uniform(b_mu, b_sigma, (output_dim, 1))
        else:
            self.weights, self.biases = initializer(
                self.input_dim, self.output_dim)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = np.dot(self.weights, input_tensor) + self.biases

        return self.output_tensor

    def backward(self, input_gradients):
        self.input_gradients = input_gradients
        self.output_gradients = np.dot(self.weights.T, input_gradients)

        return self.output_gradients

    def gradient_per_weights(self, input_gradients):
        self.dW = np.dot(input_gradients, self.input_tensor.T)
        self.dB = input_gradients.sum(axis=1, keepdims=True)

        if self.weights_regularizer:
            self.dW += self.weights_regularizer(self.weights)

        return self.dW, self.dB

    def optimize(self, weights_updates):
        self.weights += weights_updates[0]
        self.biases += weights_updates[1]
