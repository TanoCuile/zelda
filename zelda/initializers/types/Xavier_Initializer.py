import numpy as np
from .Abstract_Initializer import Abstract_Initializer


class Xavier_Initializer(Abstract_Initializer):
    def __init__(self,
                 magnitude=None,
                 factor_type=None,
                 randomization_type=None):
        super(Xavier_Initializer, self).__init__()
        self.magnitude = magnitude
        self.factor_type = factor_type
        self.randomization_type = randomization_type

    def __call__(self,
                 input_dimension,
                 output_dimension,
                 magnitude=3,
                 factor_type='avg',
                 randomization_type='uniform'):

        magnitude, factor_type, randomization_type = self._init_parameters(
            magnitude=magnitude, factor_type=factor_type, randomization_type=randomization_type)

        scale = self._get_scale(magnitude=magnitude, factor=self._get_factor(
            factor_type, input_dimension, output_dimension))

        if randomization_type == "gaussian":
            return(self._get_gausian_initialized_vectors(
                scale, output_dimension, input_dimension))

        return(self._get_uniform_initialized_vectors(
            scale, output_dimension, input_dimension))

    def _get_uniform_initialized_vectors(self, scale, output_dimension, input_dimension):
        weights = np.random.uniform(
            low=0, high=scale, size=(output_dimension, input_dimension))
        biases = np.random.uniform(
            low=0, high=scale, size=(output_dimension, 1))
        return weights, biases

    def _get_gausian_initialized_vectors(self, scale, output_dimension, input_dimension):
        weights = np.random.normal(loc=0, scale=scale, size=(
            output_dimension, input_dimension))
        biases = np.random.normal(
            loc=0, scale=scale, size=(output_dimension, 1))
        return weights, biases

    def _get_scale(self, magnitude, factor):
        scale = np.sqrt(magnitude / factor)
        return scale

    def _get_factor(self, factor_type, input_dimension, output_dimension):
        if factor_type == "in":
            return input_dimension
        elif factor_type == "out":
            return output_dimension
        else:
            return (input_dimension + output_dimension) / 2.0

    def _init_parameters(
            self,
            magnitude,
            factor_type,
            randomization_type):
        if self.magnitude is not None:
            magnitude = self.magnitude
        if self.factor_type is not None:
            factor_type = self.factor_type
        if self.randomization_type is not None:
            randomization_type = self.randomization_type

        return ((magnitude,
                 factor_type,
                 randomization_type))
