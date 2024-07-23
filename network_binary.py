import numpy as np
from nnfs.datasets import spiral_data
from activation.relu import ActivationReLU
from activation.sigmoid import SigmoidActivation
from layer.dense_layer import DenseLayer
from loss.binary_cross_entropy import BinaryCrossEntropyLoss
from optimizers.adam import AdamOptimizer

X, y = spiral_data(samples=100, classes=2)
y = y.reshape(-1, 1)

dense1 = DenseLayer(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = ActivationReLU()
dense2 = DenseLayer(64, 1)
activation2 = SigmoidActivation()

loss_function = BinaryCrossEntropyLoss()
optimizer = AdamOptimizer(decay=5e-7)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    data_loss = loss_function.calculate(activation2.output, y)
    regularization_loss = loss_function.regularization_loss(dense1) + \
                            loss_function.regularization_loss(dense2)
    loss = data_loss + regularization_loss
    
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)
    
    if not epoch % 1000:
        print(f'epoch: {epoch}, accuracy: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss}, reg_loss: {regularization_loss}), lr: {optimizer.current_learning_rate}')

    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
    
x_test, y_test = spiral_data(samples=100, classes=2)

y_test = y_test.reshape(-1, 1)

dense1.forward(x_test)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output, y_test)
predictions = (activation2.output > 0.5) * 1

y_test = np.mean(predictions == y_test)
print(f'validation, accuracy: {accuracy:.3f}, loss: {loss:.3f}')