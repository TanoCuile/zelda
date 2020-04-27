from .SGD_Optimizer import SGD_Optimizer

class optimizers:
    def __init__(self):
        pass

    class SGD:
        def __init__(self, lr):
            self.lr = lr

        def __call__(self):
            return SGD_Optimizer(self.lr)
