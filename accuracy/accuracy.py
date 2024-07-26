import numpy as np

class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        return np.mean(comparisons)
        