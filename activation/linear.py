class LinearActivation:
    def forward(self, inputs, _training):
        self.inputs = inputs
        self.output = inputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        
    def predictions(self, outputs):
        return outputs