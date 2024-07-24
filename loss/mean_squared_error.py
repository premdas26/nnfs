from loss.loss import Loss
import numpy as np


class MeanSquaredErrorLoss(Loss):
    def forward(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2, axis=-1)
    
    def backward(self, dvalues, y_true):
        num_samples = len(dvalues)
        num_outputs = len(dvalues[0])
        
        self.dinputs = -2 * (y_true - dvalues) / num_outputs
        self.dinputs = self.dinputs / num_samples