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
        for epoch in range(epochs):
            self._run_epoch(X_train, Y_train,
                            epoch_number=epoch, verbose=verbose)

    def _run_epoch(self, X, Y, epoch_number=None, verbose=False):

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
