from .Model import Model
from ..utils import utils


class DebuggableModel(Model):
    def __init__(self):
        Model.__init__(self)
        self.pbar = None

    def train(self, X_train, Y_train, epochs, verbose=False):
        if utils.is_ipython_env():
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        self.pbar = tqdm(total=epochs)
        Model.train(self, X_train=X_train, Y_train=Y_train,
                    epochs=epochs, verbose=verbose)

    def _run_epoch(self, X, Y, epoch_number, verbose=False):
        current_loss_value = Model._run_epoch(
            self, X=X, Y=Y)

        self.pbar.update(1)

        if verbose:
            if epoch_number % 10 == 0:
                print()
                self.pbar.set_description(
                    f'Epoch: {epoch_number} Loss: {current_loss_value:.4f} Progress ')

                # print(np.around(self.layers[-1].output_tensor.T[11], decimals=4))
                # print(np.around(self.layers[-1].input_gradients.T[11], decimals=9))
                # print(np.around(self.layers[-1].output_gradients.T[11], decimals=9))
