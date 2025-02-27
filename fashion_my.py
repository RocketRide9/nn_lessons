#!/usr/bin/env python

from numpy.typing import NDArray
from mnist_reader import load_mnist
import numpy as np

STEP_SIZE = 0.002

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

class Transformer:
    def forward(self, x) -> NDArray:
        assert(False)
    def backward(self, x) -> NDArray:
        assert(False)
        
class Relu(Transformer):
    prev: NDArray
    
    def forward(self, x):
        self.prev = x.copy()
        return np.maximum(x, 0)
    def backward(self, x):
        return x @ np.where(self.prev > 0, 1, 0)

class Linear(Transformer):
    layer: Layer
    prev: NDArray
    
    def __init__(self, layer: Layer):
        self.layer = layer
    
    def forward(self, x):
        self.prev = x.copy()
        return x @ self.layer.matrix + self.layer.bias
    def backward(self, x):
        dW = np.outer(x, self.prev)
        db = np.sum(x)
        self.layer.bias -= STEP_SIZE * db
        self.layer.matrix -= STEP_SIZE * dW
        return dW
 

class Model:
    transformers = []
    loss = 0
    def __init__(self):
        self.transformers.append(Linear(Layer(28*28, 512)))
        self.transformers.append(Relu())
        self.transformers.append(Linear(Layer(512, 512)))
        self.transformers.append(Relu())
        self.transformers.append(Linear(Layer(512, 10)))
    
    def run(self, x):
        for i in range(len(self.transformers)):
            x = self.transformers[i].forward(x)

        return x
    
    def optimize(self, y, y_true):
        y_soft = softmax(y)
        self.loss = cross_entropy(y_soft, y_true)
        b = diff_cel(y_soft, y_true)
        
        for i in range(len(self.transformers)-1, -1, -1):
            b = self.transformers[i].backward(b)
            
        
        
        
def diff_cel(y_soft, y_true):
    return y_soft - y_true

def relu(x):
    return np.maximum(x, 0)

def linear(layer: Layer, x):
    b = layer.bias
    a = layer.matrix
    
    return x @ a + b
    
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy(y, y_true):
    return -np.sum(y_true * np.log(y))

def loss_fn(res, label):                                
    return np.mean((res - label) ** 2)
 
def SGD(image, label, res, lr, layer: Layer):
    g_matrix = (2 / res.size) * image.T @ (res - label)       
    g_bias = (2 / res.size) * np.sum(res - label)         
    layer.matrix -= lr * g_matrix.reshape(-1, 1)                  
    layer.bias -= lr * g_bias                             

def train(imgs, labels):
    iters = int(len(imgs)/BATCH_SIZE)
    test_loss, correct = 0, 0
    for i in range(0, iters*BATCH_SIZE, BATCH_SIZE):
        image = imgs[i:i+BATCH_SIZE] / 256.0
        label = labels[i: i+BATCH_SIZE]

        res = model.run(image)

        y = np.zeros((len(label), 10), dtype=float)  # Создаем массив нулей
        y[np.arange(len(label)), label] = 1  # Вставляем единицы по индексам

        model.optimize(y, res)


        res = softmax(res)
        loss = model.loss
        test_loss += loss

        correct += (np.argmax(res, 1) == label).sum().item()

        if i % (100*BATCH_SIZE) == 0:
            #print(res- y)
            print(f"loss: {loss:>7f}  [{i:>5d}/{len(imgs):>5d}]")
    return test_loss / iters, correct / len(imgs)

BATCH_SIZE = 60

imgs, labels = load_mnist('data/FashionMNIST/raw/')
model = Model()

epochs = 5
for epoch  in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    test_loss, correct = train(imgs, labels)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
print("Done!")
