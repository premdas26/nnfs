from accuracy.accuracy_categorical import CategoricalAccuracy
from activation.softmax import ActivationSoftmax
from layer import DenseLayer, DropoutLayer
from activation import ActivationReLU
from loss.categorical_cross_entropy import CategoricalCrossentropyLoss
from model import Model
from softmax_classifier import SoftmaxClassifier
from optimizers import SGDOptimizer, AdaOptimizer, RMSPropOptimizer, AdamOptimizer
import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data

nnfs.init()

X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

model = Model()
model.add(DenseLayer(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(ActivationReLU())
model.add(DropoutLayer(0.1))
model.add(DenseLayer(512, 3))
model.add(ActivationSoftmax())

model.set(
    loss=CategoricalCrossentropyLoss(),
    optimizer=AdamOptimizer(learning_rate=0.05, decay=5e-5),
    accuracy=CategoricalAccuracy()
)

model.finalize()
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)