import numpy as np

class Utils():
    def __init__(self):
        self._is_ipython = None
        self._is_ipython = self.is_ipython_env()

    def is_ipython_env(self):
        if self._is_ipython is not None:
            return self._is_ipython

        try:
            __IPYTHON__
            return(True)
        except NameError:
            return(False)
    def flatten(self, tensor):
        tensor_shape = tensor.shape
        flatten_tensor = tensor.reshape(tensor_shape[0], np.product(tensor_shape[1:]))
        return flatten_tensor

    def one_hot(self, y):
        class_number = len(set(y))
        one_hot_matrix = np.zeros((len(y), class_number))
        for i, y_val in enumerate(y):
            one_hot_matrix[i][int(y_val)] = 1.
        return one_hot_matrix

    def preprocess_keras_image_dataset_for_dnn(self, *, x_train, y_train, x_test, y_test):
        x_train = self.flatten(x_train).astype('float32') / 255
        x_test = self.flatten(x_test).astype('float32') / 255

        y_train = self.one_hot([int(y) for y in y_train])
        y_test = self.one_hot([int(y) for y in y_test])

        return (x_train, y_train), (x_test, y_test)

utils = Utils()
