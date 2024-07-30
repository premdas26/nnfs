import cv2
import os
import nnfs
import numpy as np

from accuracy.accuracy_categorical import CategoricalAccuracy
from activation.relu import ActivationReLU
from activation.softmax import ActivationSoftmax
from layer.dense_layer import DenseLayer
from loss.categorical_cross_entropy import CategoricalCrossentropyLoss
from model import Model
from optimizers.adam import AdamOptimizer

nnfs.init()

def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))

    X, y = [], []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            
            X.append(image)
            y.append(label)
            
    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    
    return X, y, X_test, y_test

X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

model = Model()

model.add(DenseLayer(X.shape[1], 128))
model.add(ActivationReLU())
model.add(DenseLayer(128, 128))
model.add(ActivationReLU())
model.add(DenseLayer(128, 10))
model.add(ActivationSoftmax())

model.set(loss=CategoricalCrossentropyLoss(), optimizer=AdamOptimizer(decay=1e-3),
          accuracy=CategoricalAccuracy())

model.finalize()
model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)
