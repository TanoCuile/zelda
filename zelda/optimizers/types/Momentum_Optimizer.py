from .Abstract_Optimizer import Abstract_Optimizer


class Momentum_Optimizer(Abstract_Optimizer):
    def __init__(self, lr):
        Abstract_Optimizer.__init__(self, layer_name='SGD')
        self.lr = lr

    def __call__(self, list_of_gradients):
        return [-1 * self.lr * gradient for gradient in list_of_gradients]
