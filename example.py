from zelda import zelda

model = zelda.Model()

model.add(zelda.layers.Linear(784, 144))
model.add(zelda.layers.Sigmoid())

model.add(zelda.layers.Linear(144, 10))
model.add(zelda.layers.Sigmoid())

sgd = zelda.optimizers.SGD(lr=0.8)
model.compile(zelda.layers.BinaryCrossEntropy(), sgd)
