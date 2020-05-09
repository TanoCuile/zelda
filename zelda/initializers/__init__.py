from .types import Uniform_Initializer, Xavier_Initializer

class Initializers():
    def __init__(self):
        self.uniform = Uniform_Initializer.Uniform_Initializer()
        self.xavier = Xavier_Initializer.Xavier_Initializer()
        self.Xavier = Xavier_Initializer.Xavier_Initializer


initializers = Initializers()
