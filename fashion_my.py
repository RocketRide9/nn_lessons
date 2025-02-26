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
    
    def run(self, x, layer: Layer):
        y = linear(layer, x)
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

def loss_fn(res, label):                                
    return np.mean((res - label) ** 2)
 
def SGD(image, label, res, lr, layer: Layer):
    g_matrix = (2 / res.size) * image.T @ (res - label)       
    g_bias = (2 / res.size) * np.sum(res - label)         
    layer.matrix -= lr * g_matrix.reshape(-1, 1)                  
    layer.bias -= lr * g_bias                             

def train(imgs, labels, layer: Layer):
    iters = int(len(imgs)/BATCH_SIZE)
    test_loss, correct = 0, 0
    for i in range(0, iters*BATCH_SIZE, BATCH_SIZE):
        image = imgs[i:i+BATCH_SIZE] / 256.0
        label = labels[i: i+BATCH_SIZE]

        res = model.run(image, layer)
        ress = np.max(res, 1)
        res = np.argmax(res, 1)
        loss = loss_fn(ress, label)

        test_loss += loss.item()
        correct += (res == label).sum().item()

        SGD(image, label, res, 0.001, layer)
        if i % (100*BATCH_SIZE) == 0:
            loss, current = loss.item(), i
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(imgs):>5d}]")
    return test_loss / iters, correct / len(imgs)

BATCH_SIZE = 60

imgs, labels = load_mnist('data/FashionMNIST/raw/')
model = Model()
layer = Layer(28*28, 512)

epochs = 5
for epoch  in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    test_loss, correct = train(imgs, labels, layer)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
print("Done!")