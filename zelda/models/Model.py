class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.model_len = 0
        self.loss_list = []
        self.last_prediction = None

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

    def make_prediction(self, X):
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
        loss_value, loss_gradient = self._forward_propagate(X=X, Y=Y)

        # backprop
        self._back_propagate(loss_gradient=loss_gradient)

        return loss_value

    def _forward_propagate(self, X, Y):
        self.last_prediction = self.make_prediction(X)

        loss_value = self.loss.forward(self.last_prediction, Y)
        loss_gradient = self.loss.backward()

        self.loss_list.append(loss_value)

        return((loss_value, loss_gradient))

    def _back_propagate(self, loss_gradient):
        latest_gradient = loss_gradient
        for i in reversed(range(self.model_len)):
            if not self.layers[i].is_trainable():
                latest_gradient = self.layers[i].backward(latest_gradient)
            else:
                weights_gradients = self.layers[i].apply_previous_gradient(
                    latest_gradient)
                if i != 0:
                    latest_gradient = self.layers[i].backward(latest_gradient)

                weights_updates = self.layers[i].optimizer(weights_gradients)
                self.layers[i].optimize(weights_updates)
