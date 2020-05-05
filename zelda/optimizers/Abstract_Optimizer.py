class Abstract_Optimizer:
    def __init__(self, layer_name='Abstract'):
        self.layer_name = layer_name

    def __str__(self):
        return f"{self.layer_name} Optimizer"

    def __call__(self):
        pass
