# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:49:24 2019

@author: mifiamigahna
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds

def getData1():
    X, Y = ds.load_iris(True)
    return (np.hstack((np.ones([len(X), 1]), X[:, :2])), Y)

def getData2():
    X, Y = ds.load_iris(True)
    return (np.hstack((np.ones([len(X), 1]), X[:, 0:1], X[:, 3:4])), Y)


class Perceptron:
        
    def __init__(self, W = "random"):
        self.W = np.random.uniform(-1.0, 1.0, 3) if type(W) == str else W
    
    def train(self, X, Y, learnRate, epochs):
        error = []
        
        for epoch in range(epochs):
            YHat = [np.sign(x.dot(self.W)) for x in X]
            false = np.nonzero([Y[i] != YHat[i] for i in range(len(Y))])[0]
            error.append(len(false) / len(Y))
            
            for i in false:
                for j in range(len(self.W)):
                    self.W[j] += learnRate * Y[i] * X[i][j]
        return error
                
    def inClass(self, X):
        return True if np.sign(X.dot(self.W)) == 1 else False
    
    def boundary(self, X):
        return [-(self.W[0] + self.W[1] * x) / self.W[2] for x in X]
    
    
class Pocket(Perceptron):
    
    def __init__(self, W = "random"):
        self.W = np.random.uniform(-1.0, 1.0, 3) if type(W) == str else W
        self.bestW = []
    
    def train(self, X, Y, learnRate, epochs):
        error = []
        bestErr = np.inf
        bestErrs = []
        
        for epoch in range(epochs):
            YHat = [np.sign(x.dot(self.W)) for x in X]
            false = np.nonzero([Y[i] != YHat[i] for i in range(len(Y))])[0]
            
            curErr = len(false) / len(Y)
            error.append(curErr)
            if curErr < bestErr:
                bestErrs.append(curErr)
                bestErr = curErr
            else:
                bestErrs.append(bestErr)
                
            for i in false:
                for j in range(len(self.W)):
                    self.W[j] += learnRate * Y[i] * X[i][j]
        return (error, bestErrs)

#---1---

X, Y = getData1()
Y0 = [1 if y == 0 else -1 for y in Y]

[plt.scatter(X[i][1], X[i][2], c = 'G' if Y[i] == 0 else 'R') for i in range(len(X))]

#---2---

percy = Perceptron()
error = percy.train(X, Y0, 1, 50)
pltX = [4.3, 7.9]

plt.figure()
[plt.scatter(X[i][1], X[i][2], c = 'G' if percy.inClass(X[i]) else 'R') for i in range(len(X))]
plt.plot(pltX, percy.boundary(pltX), c = 'Y')

#---3---

plt.figure()
plt.plot(error, c = 'R')

#---5---

X, Y = getData2()
Y1 = [1 if y == 1 else -1 for y in Y]

plt.figure()
[plt.scatter(X[i][1], X[i][2], c = 'G' if Y[i] == 1 else 'R') for i in range(len(X))]

#---6---

poketmon = Pocket()
error, bestErrs = poketmon.train(X, Y1, 1, 50)

plt.figure()
plt.plot(error, c = 'R')
plt.plot(bestErrs, c = 'G')