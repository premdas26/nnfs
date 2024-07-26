import numpy as np
from nnfs.datasets import spiral_data
from accuracy.accuracy_categorical import CategoricalAccuracy
from activation.relu import ActivationReLU
from activation.sigmoid import SigmoidActivation
from layer.dense_layer import DenseLayer
from loss.binary_cross_entropy import BinaryCrossEntropyLoss
from model import Model
from optimizers.adam import AdamOptimizer

X, y = spiral_data(samples=100, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

y = y.reshape(-1, 1)
y_test = y.reshape(-1, 1)

model = Model()

model.add(DenseLayer(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(ActivationReLU())
model.add(DenseLayer(64, 1))
model.add(SigmoidActivation())

model.set(loss=BinaryCrossEntropyLoss(), optimizer=AdamOptimizer(decay=5e-7), accuracy=CategoricalAccuracy(binary=True))
model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)