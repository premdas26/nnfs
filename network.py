from dense_layer import DenseLayer
from activation import ActivationReLU, ActivationSoftmax
from loss import CategoricalCrossentropyLoss
import math
import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data

nnfs.init()

# x, y = spiral_data(samples=100, classes=3)
x, y = vertical_data(samples=100, classes=3)

dense1 = DenseLayer(2, 3)
activation1 = ActivationReLU()

dense2 = DenseLayer(3, 3)
activation2 = ActivationSoftmax()

dense1.forward(x)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_calculator = CategoricalCrossentropyLoss()
loss = loss_calculator.calculate(activation2.output, y)

lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()

best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration  in range(10000):
    dense1.weights = 0.05 * np.random.randn(2, 3)
    dense1.biases = 0.05 * np.random.randn(1, 3)
    
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases = 0.05 * np.random.randn(1, 3)
    
    dense1.forward(x)
    activation1.forward(dense1.output)
    
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    loss = loss_calculator.calculate(activation2.output, y)
    
    

