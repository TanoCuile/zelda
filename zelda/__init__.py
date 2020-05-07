from .models import Model
from .optimizers import optimizers
from .layers import layers
from .utils import utils


class Zelda():
    def __init__(self):
        self.Model = Model.Model
        # if utils.is_ipython_env():
        #     from .models.DebuggableModel import DebuggableModel
        #     self.DebuggableModel = DebuggableModel
        # else:
        from .models.DebuggableModel import DebuggableModel
        self.DebuggableModel = DebuggableModel

        self.optimizers = optimizers
        self.layers = layers

        self.utils = utils


zelda = Zelda()
