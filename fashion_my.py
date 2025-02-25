#!/usr/bin/env python

from mnist_reader import load_mnist
import numpy as np

class Layer:
    matrix = None
    bias = None
    
    # матрица mxn
    # m - выходное количество нейронов
    # n - входное количество нейронов
    def __init__(self, m, n):
        assert (m > 0 and n > 0)
        bound = np.sqrt(1/m)
        self.matrix = np.random.uniform(-bound, bound, (m, n))
        self.bias = np.random.uniform(-bound, bound, n)

class Model:
    layers = []
    
    def __init__(self):
        self.layers.append(Layer(28*28, 512))
        self.layers.append(Layer(512, 512))
        self.layers.append(Layer(512, 10))
    
    def run(self, x):
        y = linear(self.layers[0], x)
        y = relu(y)
        
        y = linear(self.layers[1], y)
        y = relu(y)
        
        y = linear(self.layers[2], y)
        return y
        

def relu(x):
    return np.maximum(x, 0)

def linear(layer: Layer, x):
    b = layer.bias
    a = layer.matrix
    
    return x @ a + b
    
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

BATCH_SIZE = 10

imgs, labels = load_mnist('data/FashionMNIST/raw/')
model = Model()

image = imgs[0:BATCH_SIZE] / 256.0
res = model.run(image)
res = softmax(res)
print("result:", np.argmax(res, 1))
print("label:", labels[0:BATCH_SIZE])
