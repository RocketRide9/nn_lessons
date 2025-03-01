#!/usr/bin/env python

from numpy.typing import NDArray
from mnist_reader import load_mnist
import numpy as np

STEP_SIZE = 0.002
BATCH_SIZE = 60

class Layer:
    matrix = None
    bias = None
    matrix_grad = None
    bias_grad = None
    
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
        return np.where(self.prev > 0, 1, 0) * x

class Linear(Transformer):
    layer: Layer
    prev: NDArray
    
    def __init__(self, layer: Layer):
        self.layer = layer
    
    def forward(self, x):
        self.prev = x.copy()
        return x @ self.layer.matrix + self.layer.bias
    def backward(self, x):
        dW = np.einsum("bi,bj->ij", self.prev, x)
        db = np.sum(x, axis=0) / BATCH_SIZE

        dY = np.sum(x @ self.layer.matrix.T, 0) / BATCH_SIZE
        self.layer.bias -= STEP_SIZE * db
        self.layer.matrix -= STEP_SIZE * dW
        return dY
 

class Model:
    transformers = []
    loss = 0
    result = []
    def __init__(self):
        self.transformers.append(Linear(Layer(28*28, 512)))
        self.transformers.append(Relu())
        self.transformers.append(Linear(Layer(512, 512)))
        self.transformers.append(Relu())
        self.transformers.append(Linear(Layer(512, 10)))
    
    def run(self, x):
        for i in range(len(self.transformers)):
            x = self.transformers[i].forward(x)
        self.result = softmax(x)
        return self.result
    
    def optimize(self, y_true):
        self.loss = np.average(cross_entropy(self.result, y_true))

        dZ = diff_cel(self.result, y_true)
        
        for i in range(len(self.transformers)-1, -1, -1):
            dZ = self.transformers[i].backward(dZ)
        
def diff_cel(y_pred, y_true):
    return y_pred - y_true

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), 1)[:, np.newaxis]

def cross_entropy(y, y_true):
    return -np.sum(y_true * np.log(y), 1)

def train(imgs, labels):
    iters = int(len(imgs) / BATCH_SIZE)
    loss_sum, correct = 0, 0
    for i in range(0, iters*BATCH_SIZE, BATCH_SIZE):
        img_batch = imgs[i:i+BATCH_SIZE] / 256.0
        label_batch = labels[i: i+BATCH_SIZE]

        res = model.run(img_batch)

        y = np.zeros((len(label_batch), 10), dtype=float)  # Создаем массив нулей
        y[np.arange(len(label_batch)), label_batch] = 1  # Вставляем единицы по индексам
        model.optimize(y)

        loss = model.loss
        loss_sum += loss

        correct += (np.argmax(res, 1) == label_batch).sum().item()

        if i % (100*BATCH_SIZE) == 0:
            print(f"loss: {loss}  {i}/{len(imgs)}")
            #print(f"loss: {loss:>7f}  [{i:>5d}/{len(imgs):>5d}]")
    return loss_sum / iters, correct / len(imgs)



imgs, labels = load_mnist('data/FashionMNIST/raw/')
model = Model()

epochs = 5
for epoch  in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    test_loss, correct = train(imgs, labels)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
print("Done!")
