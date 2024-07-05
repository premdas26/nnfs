from dense_layer import DenseLayer
from activation import ActivationReLU
from softmax_classifier import SoftmaxClassifier
from optimizers import SGDOptimizer, AdaOptimizer, RMSPropOptimizer, AdamOptimizer
import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data

nnfs.init()

x, y = spiral_data(samples=100, classes=3)

dense1 = DenseLayer(2, 64)
activation1 = ActivationReLU()

dense2 = DenseLayer(64, 3)
loss_activation = SoftmaxClassifier()

# optimizer = SGDOptimizer(decay=1e-3, momentum=0.88)
# optimizer = AdaOptimizer(decay=1e-4)
# optimizer = RMSPropOptimizer(decay=1e-5, learning_rate=0.02, rho=0.999)
optimizer = AdamOptimizer(learning_rate=0.05, decay=5e-7)

for epoch in range(10001):
    dense1.forward(x)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    
    if not epoch % 100:
        print(f'epoch: {epoch}, accuracy: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
