import numpy as np
from nnfs.datasets import sine_data

from accuracy.accuracy_regression import RegressionAccuracy
from activation.linear import LinearActivation
from activation.relu import ActivationReLU
from layer.dense_layer import DenseLayer
from loss.mean_squared_error import MeanSquaredErrorLoss
from optimizers.adam import AdamOptimizer
import matplotlib.pyplot as plt
from model import Model

X, y = sine_data()

model = Model()
model.add(DenseLayer(1, 64))
model.add(ActivationReLU())
model.add(DenseLayer(64, 64))
model.add(ActivationReLU())
model.add(DenseLayer(64, 1))
model.add(LinearActivation())

model.set(
    loss=MeanSquaredErrorLoss(), 
    optimizer=AdamOptimizer(learning_rate=0.005, decay=1e-3),
    accuracy=RegressionAccuracy()
)
model.finalize()

model.train(X, y, epochs=10000, print_every=100)