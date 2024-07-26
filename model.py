from activation.softmax import ActivationSoftmax
from layer.input_layer import InputLayer
from loss.categorical_cross_entropy import CategoricalCrossentropyLoss
from softmax_classifier import SoftmaxClassifier


class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None
    
    def add(self, layer):
        self.layers.append(layer)
        

    def set(self, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
        
    def train(self, X, y, epochs=1, print_every=1, validation_data=None):
        self.accuracy.init(y)
        
        for epoch in range(1, epochs + 1):
            output = self.forward(X, training=True)
            
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss
            
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)
                
            self.backward(output, y)
            
            self.optimizer.pre_update_params()
            for layer in self.loss.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
            
            if not epoch % print_every:
                print(f'epoch: {epoch}, loss: {loss:.3f},'
                      f'accuracy: {accuracy:.3f}, data_loss: {data_loss:.3f},'
                      f'reg_loss: {regularization_loss:.3f}, lr: {self.optimizer.current_learning_rate}')
        
        if validation_data is not None:
            X_val, y_val = validation_data
            output = self.forward(X_val, training=False)
            
            loss = self.loss.calculate(output, y_val)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)
            print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
            
        
    def finalize(self):
        self.input_layer = InputLayer()
        layer_count = len(self.layers)
        
        self.trainable_layers = []
        
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
                
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
                
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers)
        
        if isinstance(self.layers[-1], ActivationSoftmax) and isinstance(self.loss, CategoricalCrossentropyLoss):
            self.softmax_classifier_output = SoftmaxClassifier()
    
    def forward(self, X, training):
        self.input_layer.forward(X, training)
        
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        return layer.output
    
    def backward(self, output, y):
        if self.softmax_classifier_output:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
                
            return
            
        self.loss.backward(output, y)
        
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    