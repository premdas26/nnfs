import numpy as np

class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        
        return np.mean(comparisons)
    
    def calculate_accumulated(self):
        return self.accumulated_sum / self.accumulated_count
        
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
        