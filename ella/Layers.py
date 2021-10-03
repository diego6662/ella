import numpy as np
from ella.Activations import step


class Single_neuron ():


    def __init__(self, units, activation, weights = None, bias = None):
        self.units = units
        self.input_shape = 0
        self.activation = activation
        self.weights = weights
        self.bias = bias
    
    def set_weights(self, weights = None):
        if self.weights is None:
            if weights:
                self.weights = weights 
            else:
                self.weights = 2 * np.random.rand(self.units, self.input_shape) - 1 

    def set_bias(self, bias = None): 
        if self.bias is None:
            if bias:
                self.bias = bias 
            else:
                self.bias = 2 * np.random.rand(self.units) - 1

    def set_input_shape (self, input_shape):
        self.input_shape = input_shape[0]

    def compute(self,X):
        value = np.dot(self.weights, X) + self.bias
        return self.activation.run(value)


class Perceptron ():

    def __init__(self, units, activation, weights = None, bias = None, learning_rate = 0.01):
        self.units = units
        self.activation = activation
        self.weights = weights
        self.bias = bias
        self.input_shape = None
        self.learning_rate = learning_rate

    def set_weights(self, weights = None):
        if self.weights is None:
            if weights:
                self.weights = weights 
            else:
                self.weights = 2 * np.random.rand(self.units, self.input_shape) - 1 

    def set_bias(self, bias = None): 
        if self.bias is None:
            if bias:
                self.bias = bias 
            else:
                self.bias = 2 * np.random.rand(self.units) - 1

    def set_input_shape (self, input_shape):
        self.input_shape = input_shape[0]    

    def set_input_shape_train (self, input_shape):
        self.input_shape = input_shape[1] 
    
    def compute(self,X):
        value = np.dot(self.weights, X) + self.bias
        return self.activation.run(value)
    
    def update_weights(self, error, x_i):
        self.weights += self.learning_rate * error * x_i
        self.bias += self.learning_rate * error

    def call(self,X, y, epochs,):
        history = []
        for epoch in range(epochs):
            error_it = 0.0
            for idx, x in enumerate(X):
                y_hat = self.compute(x)
                error = (y[idx] - y_hat)
                error = np.mean(error)
                error_it += error
                if error != 0:
                    self.update_weights(error, x)
            history.append(error_it)
        return history

class Adeline ():

    def __init__(self, units, activation, weights = None, bias = None, learning_rate = 0.01, tao = 1e-4):
        self.units = units
        self.activation = activation
        self.weights = weights
        self.bias = bias
        self.input_shape = None
        self.learning_rate = learning_rate
        self.tao = tao

    def set_weights(self, weights = None):
        if self.weights is None:
            if weights:
                self.weights = weights 
            else:
                self.weights = 2 * np.random.rand(self.units, self.input_shape) - 1 

    def set_bias(self, bias = None): 
        if self.bias is None:
            if bias:
                self.bias = bias 
            else:
                self.bias = 2 * np.random.rand(self.units) - 1

    def set_input_shape (self, input_shape):
        self.input_shape = input_shape[0]

    def set_input_shape_train (self, input_shape):
        self.input_shape = input_shape[1]
    
    def compute(self,X):
        value = np.dot(self.weights, X) + self.bias
        return self.activation.run(value)
    
    def update_weights(self, error, x_i):
        self.weights += self.learning_rate * error * x_i
        self.bias += self.learning_rate * error

    def call(self,X, y, epochs, ):
        history = []
        for epoch in range(epochs):
            error_it = 0.0
            for idx, x in enumerate(X):
                y_hat = self.compute(x)
                error = (y[idx] - y_hat)
                error = np.mean(error)
                error_it += (error ** 2) / 2.0
                self.update_weights(error, x)

            history.append(error_it)
            if error_it <= self.tao:
                break
        return history

   
