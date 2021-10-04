import numpy as np
from tqdm import tqdm


class Single_sequential ():
    """ only feed forward """    

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
        history = []
        for epoch in tqdm(range(epochs)):
            error = 0.0
            value = X.copy()
            input_shape = value.shape
            for layer in self.layers[:-1]:
                layer.set_input_shape_train(input_shape)
                layer.set_weights()
                layer.set_bias()
                layer.call(value,y)
                value = np.array([layer.compute(v) for v in value])
                input_shape = value.shape
            layer = self.layers[-1]
            layer.set_input_shape_train(input_shape)
            layer.set_weights()
            layer.set_bias()
            error = layer.call(value,y)
            history.append(error)
        return history

    def predict(self,X):
        result = []
        for x in X:
            value = self.run(x)
            result.append(value)
        return np.array(result)
        

class Sequential ():
    """ backpropagation """    

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
 
