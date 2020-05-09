import numpy as np
from .Abstract_Initializer import Abstract_Initializer


class Uniform_Initializer(Abstract_Initializer):
    def __call__(self,
                 input_dimension,
                 output_dimension,
                 weights_mu=0.,
                 weights_sigma=0.01,
                 biases_mu=0.0001,
                 biases_sigma=0.005):
        weights = np.random.uniform(
            weights_mu, weights_sigma, (output_dimension, input_dimension))
        biases = np.random.uniform(
            biases_mu, biases_sigma, (output_dimension, 1))

        return((weights, biases))
