import numpy as np
from loss.loss import Loss

class MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        return np.mean(np.abs(y_true - y_pred), axis=-1)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        
        outputs = len(dvalues[0])
        
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples