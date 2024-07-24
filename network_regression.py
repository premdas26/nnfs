import numpy as np
from nnfs.datasets import sine_data

from activation.linear import LinearActivation
from activation.relu import ActivationReLU
from layer.dense_layer import DenseLayer
from loss.mean_squared_error import MeanSquaredErrorLoss
from optimizers.adam import AdamOptimizer
import matplotlib.pyplot as plt

X, y = sine_data()

dense1 = DenseLayer(1, 64)
activation1 = ActivationReLU()

dense2 = DenseLayer(64, 64)
activation2 = ActivationReLU()

dense3 = DenseLayer(64, 1)
activation3 = LinearActivation()

loss_function = MeanSquaredErrorLoss()
optimizer = AdamOptimizer(learning_rate=0.005, decay=1e-3)

accuracy_precision = np.std(y) / 250

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    
    data_loss = loss_function.calculate(activation3.output, y)
    regularization_loss = loss_function.regularization_loss(dense1) \
                            + loss_function.regularization_loss(dense2) \
                            + loss_function.regularization_loss(dense3)
    loss = data_loss + regularization_loss
    
    predictions = activation3.output
    accuracy = np.mean(np.abs(activation3.output - y) < accuracy_precision)
    
    if not epoch % 100:
        print(f'epoch: {epoch}, acc: {accuracy}, loss: {loss}')
        
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
    

X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

dense3.forward(activation2.output)
activation3.forward(dense3.output)

plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)
plt.show()