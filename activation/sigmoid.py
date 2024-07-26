import numpy as np

class SigmoidActivation:
    def forward(self, inputs, _training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
        
    def predictions(self, outputs):
        return (outputs > 0.5) * 1