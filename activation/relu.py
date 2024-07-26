import numpy as np

class ActivationReLU:
    def forward(self, inputs, _training):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        
    def predictions(self, outputs):
        return outputs