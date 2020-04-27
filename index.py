from Model import Model
from optimizers import optimizers
from layers import layers

model = Model()

model.add(layers.Linear(784, 144))
model.add(layers.Sigmoid())

model.add(layers.Linear(144, 10))
model.add(layers.Sigmoid())

sgd = optimizers.SGD(lr=0.8)
model.compile(layers.BinaryCrossEntropy(), sgd)
