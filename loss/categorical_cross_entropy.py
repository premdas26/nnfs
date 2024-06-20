import numpy as np
from . import Loss

class CategoricalCrossentropyLoss(Loss):
    def forward(self, y_pred, y_true):
        num_samples = len(y_pred)
        
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            confidences = y_pred_clipped[range(num_samples), y_true]
        elif len(y_true.shape) == 2:
            confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
        return -np.log(confidences)