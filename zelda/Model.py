from tqdm.notebook import trange, tqdm


class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.model_len = 0
        self.loss_list = []
        self.predictions = None

    def __len__(self):
        return self.model_len

    def add(self, layer):
        self.layers.append(layer)
        self.model_len += 1

    def compile(self, loss, optimizer, metrics=['accuracy']):
        self.loss = loss
        self.optimizer = optimizer
        for i in range(self.model_len):
            if self.layers[i].is_trainable():
                self.layers[i].optimizer = optimizer()

    def predict(self, X):
        self.tensor = X.T
        for i in range(self.model_len):
            self.tensor = self.layers[i].forward(self.tensor)

        return self.tensor

    def train(self, X_train, Y_train, epochs, verbose=False):

        pbar = tqdm(total=epochs)
        for epoch in range(epochs):
            current_loss_value = self._run_epoch(X_train, Y_train)

            pbar.update(1)

            if verbose:
                if epoch % 10 == 0:
                    print()
                    pbar.set_description(
                        f'Epoch: {epoch} Loss: {current_loss_value:.4f} Progress ')

                    # print(np.around(self.layers[-1].output_tensor.T[11], decimals=4))
                    # print(np.around(self.layers[-1].input_gradients.T[11], decimals=9))
                    # print(np.around(self.layers[-1].output_gradients.T[11], decimals=9))

    def _run_epoch(self, X, Y):

        # forwardprop
        self.predictions = self.predict(X)

        loss_value = self.loss.forward(self.predictions, Y)
        gradient = self.loss.backward()

        self.loss_list.append(loss_value)

        # backprop
        for i in reversed(range(self.model_len)):
            if not self.layers[i].is_trainable():
                gradient = self.layers[i].backward(gradient)
            else:
                weights_gradients = self.layers[i].gradient_per_weights(
                    gradient)
                if i != 0:
                    gradient = self.layers[i].backward(gradient)

                weights_updates = self.layers[i].optimizer(weights_gradients)
                self.layers[i].optimize(weights_updates)

        return loss_value
