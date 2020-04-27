class Abstract_Layer:
    def __init__(self, is_trainable=False, layer_name='Abstract'):
        self.__is_trainable = is_trainable
        self.layer_name = layer_name

    def __str__(self):
        name = self.get_name()
        return f"{name} Layer"

    def forward(self):
        pass

    def backward(self):
        pass

    def optimizer(self):
        pass

    def optimize(self):
        pass

    def is_trainable(self):
        return(self.__is_trainable)

    def get_name(self):
        return(self.layer_name)
