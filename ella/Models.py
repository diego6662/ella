import numpy as np


class Single_sequential ():
    

    def __init__(self, layers = []):
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)
    
    def run(self,X):
        value = X.copy()
        input_shape = X.shape
        for layer in self.layers:
            layer.set_input_shape(input_shape)
            layer.set_weights()
            layer.set_bias()
            value = layer.compute(value)
            input_shape = value.shape
        return value
    
    def train(self,X, y, epochs):
        value = X.copy()
        input_shape = value.shape
        for layer in self.layers:
            layer.set_input_shape_train(input_shape)
            layer.set_weights()
            layer.set_bias()
            history = layer.call(value,y,epochs)
            value = np.array([layer.compute(v) for v in value])
            input_shape = value.shape
        return history

    def predict(self,X):
        result = []
        for x in X:
            value = self.run(x)
            result.append(value)
        return np.array(result)
        
