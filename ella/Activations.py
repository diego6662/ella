import numpy as np

@np.vectorize
def linear_func(X):
    return X

@np.vectorize
def step_func(t,X):
    if X >= t:
        return 1.0
    else:
        return 0.0

@np.vectorize
def sigmoid_func (X):
    if X < -5:
        return 0.0
    elif X > 5:
        return 1.0
    return 1.0 / (1.0 + np.exp(-X))

@np.vectorize
def tanH (X):
    return np.tanh(X)

@np.vectorize
def relu (X):
    return max(X,0)

@np.vectorize
def leaky_relu (X):
    if X > 0:
        return X
    else:
        return 0.01 * X

class step():
    def __init__(self,t = 0.5):
        self.t = t

    def run (self,X):
        return step_func(self.t,X)     

class sigmoid():
    def __init__(self,):
        pass

    def run (self,X):
        return sigmoid_func(X)     

class linear ():
    def __init__(self):
        pass

    def run(self,X):
        return linear_func(X)
