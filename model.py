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
        
    def train(self, X, y, epochs=1, batch_size=None, print_every=1, validation_data=None):
        self.accuracy.init(y)
        
        train_steps = 1
        if validation_data is not None:
            validation_steps = 1
            
            X_val, y_val = validation_data
        
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
                
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1
        
        for epoch in range(1, epochs + 1):
            print(f'epoch: {epoch}')
            
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
            
                output = self.forward(batch_X, training=True)
                
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
                    
                self.backward(output, batch_y)
                
                self.optimizer.pre_update_params()
                for layer in self.loss.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, loss: {loss:.3f},'
                        f'accuracy: {accuracy:.3f}, data_loss: {data_loss:.3f},'
                        f'reg_loss: {regularization_loss:.3f}, lr: {self.optimizer.current_learning_rate}')
                    
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            print(f'training, ', f'acc: {epoch_accuracy:.3f}, ', f'loss: {epoch_loss: .3f}, ', 
                  f'data_loss: {epoch_data_loss:.3f}, ', f'regularization_loss:{epoch_regularization_loss}',
                  f'lr: {self.optimizer.current_learning_rate}')
        
        if validation_data is not None:
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            for step in range(validation_steps):
                if batch_size is None:
                    batch_X = X_val
                    batch_y = y_val
                else:
                    batch_X = X_val[step*batch_size:(step + 1)*batch_size]
                    batch_y = y_val[step*batch_size:(step + 1)*batch_size]
                    
                output = self.forward(X_val, training=False)
                self.loss.calculate(output, y_val)
                
                predictions = self.output_layer_activation.predictions(output)
                self.accuracy.calculate(predictions, y_val)
                
            validation_loss = self.loss.calculate_accumulated()
            validation_accuracy = self.accuracy.calculate_accumulated()
            
            print(f'validation, acc: {validation_accuracy:.3f}, loss: {validation_loss:.3f}')
            
        
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
    