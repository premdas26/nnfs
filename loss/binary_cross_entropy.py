import numpy as np
from . import Loss

class BinaryCrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + 
                            (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(sample_losses, axis=-1)
    
    def backward(self, dvalues, y_true):
        num_samples = len(dvalues)
        num_outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / num_outputs
        self.dinputs = dinputs / num_samples