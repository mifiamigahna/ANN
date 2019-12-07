# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:10:43 2019

@author: mifiamigahna
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import sklearn.model_selection as ms

def getData1(n):
    x1 = (np.random.random(n) * 2 - 1).reshape(n, 1)
    x2 = (np.random.random(n) * 2 - 1).reshape(n, 1)
    
    X = np.hstack((x1, x2))
    Y = [1 if np.sign(X[i][0]) == np.sign(X[i][1]) else -1 for i in range(len(X))]
    
    return X, Y


def getData2():
    X, Y = ds.load_iris(True)
    return (np.hstack((np.ones([len(X), 1]), X[:, 0:1], X[:, 3:4])), Y)


class MLP:
    
    def __init__(self, shape = [2, 2, 1], xaver = False):
        self.shape = np.array(shape)
        self.layers = len(self.shape)
        if xaver:
            xav = np.sqrt(6 / (self.shape[0] + self.shape[-1]))
            self.W = [np.random.uniform(-xav, xav, (self.shape[layer + 1], self.shape[layer] + 1)) for layer in range(self.layers - 1)]
        else:
            self.W = [np.random.uniform(-1.0, 1.0, (self.shape[layer + 1], self.shape[layer] + 1)) for layer in range(self.layers - 1)]
        self.a = [np.zeros(self.shape[i]) for i in range(self.layers)]
        self.h = [np.zeros(self.shape[i]) for i in range(self.layers)]
        self.d = [np.zeros(self.shape[i]) for i in range(self.layers)]
        
    def init_weights(self):
        self.W = [np.random.uniform(-1.0, 1.0, (self.shape[layer + 1], self.shape[layer] + 1)) for layer in range(self.layers - 1)]
        
    def tanh(self, x):
        return  (np.e ** x - np.e ** -x) / (np.e ** x + np.e ** -x)
    
    def tanhD(self, x):
        return 1 - self.tanh(x) ** 2
    
    def add_bias(self, x):
        return np.concatenate(([1], x))
        
    def feed_forward(self, x):
        self.h[0] = self.add_bias(x)
        for layer in range(1, self.layers):
            for neuron in range(self.shape[layer]):
                self.a[layer][neuron] = self.h[layer - 1].dot(self.W[layer - 1][neuron])
            self.h[layer] = self.add_bias(self.tanh(self.a[layer]))
        return self.h[-1][-1]
    
    def predict(self, X, Y):
        YHat = np.zeros(len(X))
        error = np.zeros(len(X))
        errs = 0
        for i, x in enumerate(X):
            self.h[0] = self.add_bias(x)
            for layer in range(1, self.layers):
                for neuron in range(self.shape[layer]):
                    self.a[layer][neuron] = self.h[layer - 1].dot(self.W[layer - 1][neuron])
                self.h[layer] = self.add_bias(self.tanh(self.a[layer]))
            YHat[i] = np.sign(self.h[-1][-1])
            error[i] = (Y[i] - YHat[i]) ** 2
            errs += 1 if Y[i] != YHat[i] else 0
        return YHat, np.mean(error), errs
    
    def train(self, X, Y, learnRate, epochs):
        for epoch in range(epochs):
            obs = np.random.randint(0, len(Y))
            yHat = self.feed_forward(X[obs])
            
            self.d[-1][-1] = self.tanhD(self.a[-1][-1]) * -2 * (Y[obs] - yHat)  #error term for last layer 
            for layer in range(2, self.layers):                                 #calculate error terms
                for neuron in range(self.shape[-layer]):
                    sum_err = 0                        
                    for next_neuron in range(self.shape[-layer + 1]):
                        sum_err += self.d[-layer + 1][next_neuron] * self.W[-layer + 1][next_neuron][neuron]
                    self.d[-layer][neuron] = self.tanhD(self.a[-layer][neuron]) * sum_err
                    
            for layer in range(self.layers - 1):                                #update weights
                for neuron in range(self.shape[layer]):
                    for next_neuron in range(self.shape[layer + 1]):
                        self.W[layer][next_neuron][neuron] -= learnRate * self.d[layer + 1][next_neuron] * self.h[layer][neuron]


X, Y = getData1(10000)
#X, Y = getData2()
#Y1 = [1 if y == 1 else -1 for y in Y]
trainX, testX, trainY, testY = ms.train_test_split(X, Y, train_size = 0.8)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = ['G' if Y[i] == 1 else 'R' for i in range(len(X))], s = 3)

mses = np.zeros(100)
erors = np.zeros(100)
for run in range(100):
    ann = MLP([2, 64, 1])
    ann.train(trainX, trainY, 0.005, 5)
    YHat, mse, errs = ann.predict(testX, testY)
    mses[run] = mse
    erors[run] = errs
print(f"gemeines TESTosteron: {np.mean(mses)}")
print(f"gemeine erors: {np.mean(erors)})")
suum = 0
for i in range(100):
    suum += (mses[i] - np.mean(mses)) ** 2
suum /= len(mses)
print(f"Zoom Zoom: {suum}")

mses = np.zeros(100)
erors = np.zeros(100)
for run in range(100):
    ann = MLP([2, 64, 1], True)
    ann.train(trainX, trainY, 0.005, 1)
    YHat, mse, errs = ann.predict(testX, testY)
    mses[run] = mse
    erors[run] = errs
print(f"gemeines TESTosteron: {np.mean(mses)}")
print(f"gemeine erors: {np.mean(erors)})")
suum = 0
for i in range(100):
    suum += (mses[i] - np.mean(mses)) ** 2
suum /= len(mses)
print(f"Zoom Zoom: {suum}")

#plt.figure()
#plt.scatter(trainX[:, 0], trainX[:, 1], c = ['G' if y == 1 else 'R' for y in YHat], s = 3)
#
#plt.figure()
#plt.scatter(testX[:, 0], testX[:, 1], c = ['G' if y == 1 else 'R' for y in YHat], s = 3)