from ella.Activations import step,sigmoid,linear
from ella.Models import Single_sequential
from ella.Layers import Single_neuron, Perceptron, Adeline
import numpy as np
import random


model1 = Single_sequential()
model1.add(
        Single_neuron(2,step(t = 0.5))
        )
model2 = Single_sequential()
model2.add(
        Single_neuron(2,sigmoid())
        )
X = np.array([
    [1,1,1,0],
    [1,1,1,1],
    [1,1,0,0],
    [1,1,0,1],

    [1,0,1,0],
    [1,0,1,1],
    [1,0,0,0],
    [1,0,0,1],

    [0,1,1,0],
    [0,1,1,1],
    [0,1,0,0],
    [0,1,0,1],

    [0,0,1,0],
    [0,0,1,1],
    [0,0,0,0],
    [0,0,0,1],

])
y = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
print("punto 1.1")
print(f"y {y}")
y_hat1 = model1.predict(X)
y_hat2 = model2.predict(X)

print(f"y_hat step\n {y_hat1}")
print(f"y_hat sigmoid\n {y_hat2}")
print("punto 1.2")
model3 = Single_sequential()
model3.add(
        Single_neuron(3,step())
        )
model4 = Single_sequential()
model4.add(
        Single_neuron(3,sigmoid())
        )
y_hat3 = model3.predict(X)
y_hat4 = model4.predict(X)

print(f"y_hat step\n {y_hat3}")
print(f"y_hat sigmoid\n {y_hat4}")
print("punto 1.3")
model5 = Single_sequential()
model5.add(
        Single_neuron(3,step())
        )
model5.add(
        Single_neuron(1,step())
        )

model6 = Single_sequential()
model6.add(
        Single_neuron(3,sigmoid())
        )
model6.add(
        Single_neuron(1,sigmoid())
        )
y_hat5 = model5.predict(X)
y_hat6 = model6.predict(X)

print(f"y_hat step\n {y_hat5}")
print(f"y_hat sigmoid\n {y_hat6}")
print("punto 2.1")
# (X or Y) and X
X = np.array([
    [1,1],
    [1,0],
    [0,1],
    [0,0]
    ])

y = [ int((x[0] or x[1]) and x[0]) for x in X]
y = np.array(y)

model_7 = Single_sequential([
    Single_neuron(1,step(t = 0.5), weights = np.array([[0.4,0.1]]), bias = np.array([0.1])),
    ])

print(f"y {y}")

y_hat7 = model_7.predict(X)

print(f"y_hat\n{y_hat7}")
print("punto 2.2")
# (X and Y) or not(X and Y)
y = [ int((x[0] and x[1]) or  (not (x[0] and x[1]))) for x in X]
y = np.array(y)

model_8 = Single_sequential([
    Single_neuron(1,step(t = 0.5), weights = np.array([[0.5,0.5]]), bias = np.array([0.5])),
    ])

print(f"y {y}")

y_hat8 = model_8.predict(X)


print(f"y_hat\n{y_hat8}")
print("punto 3")
model_9 = Single_sequential([
    Perceptron(1,sigmoid(), learning_rate = 0.05),
    Perceptron(1,step(), learning_rate = 0.05),
    ])
x = np.array([[1,1,1],[1,1,0],[1,0,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]])
y = np.array([1,1,1,0,1,0,0,0])
print(y)
history = model_9.train(x,y,100)
y_hat = model_9.predict(x)
print(y_hat)
print("punto 4.1")
x = np.array([
    (random.randrange(0,30),
    random.randrange(0,20),
    random.randrange(0,10)) for i in range(30)
    ])

y = np.array([
    2 * x + 3 * y - 2 * x for x,y,z in x
    ])

model_10 = Single_sequential([
    Adeline(1,linear(), learning_rate = 0.00005)
    ])

history = model_10.train(x, y, 50000)

y_hat = model_10.predict(x)

print(y)

print(y_hat)

print("punto 4.2")

y = np.array([
    8 * x - 10 * y - 10 * z for x,y,z in x
    ])

model_11 = Single_sequential([
    Adeline(1,linear(), learning_rate = 0.001)
    ])

history = model_11.train(x, y, 50000)

y_hat = model_11.predict(x)

print(y)

print(y_hat)


