from zelda import zelda
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10

model = zelda.DebuggableModel()

model.add(zelda.layers.Linear(input_dimension=784, output_dimension=144, initializer=zelda.initializers.xavier))
model.add(zelda.layers.Sigmoid())

model.add(zelda.layers.Linear(input_dimension=144, output_dimension=10, initializer=zelda.initializers.xavier))
model.add(zelda.layers.Sigmoid())

sgd = zelda.optimizers.SGD(lr=0.8)
model.compile(loss=zelda.layers.BinaryCrossEntropy(), optimizer=sgd)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

(X_train_cifar, Y_train_cifar), (X_test_cifar, Y_test_cifar) = zelda.utils.preprocess_keras_image_dataset_for_dnn(
    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
    )

del x_train, y_train, x_test, y_test

(x_train, y_train), (x_test, y_test) = mnist.load_data()

(X_train_mnist, Y_train_mnist), (X_test_mnist, Y_test_mnist) = zelda.utils.preprocess_keras_image_dataset_for_dnn(
    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
    )

del x_train, y_train, x_test, y_test

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

(X_train_fashion_mnist, Y_train_fashion_mnist), (X_test_fashion_mnist, Y_test_fashion_mnist) = zelda.utils.preprocess_keras_image_dataset_for_dnn(
    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
    )

del x_train, y_train, x_test, y_test

model.train(X_train = X_train_mnist,
            Y_train = Y_train_mnist,
            epochs=1401,
            verbose=True)
