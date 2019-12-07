# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 21:11:35 2019

@author: mifiamigahna
"""

import numpy as np
import scipy.stats as fuck
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.load("C:\\Users\\mifiamigahna\\documents\\05_log_regression_data.npy")

x1 = data.T[0]
x2 = data.T[1]
y = data.T[2]

#sX1 = fuck.zscore(x1)
#sX2 = fuck.zscore(x2)
#sDataX = np.array([np.ones(len(sX1)),sX1, sX2]).T

stack = np.vstack((x1, x2)).T
stack = fuck.zscore(stack)

x1 = stack.T[0]
x2 = stack.T[1]

data = np.array([x1, x2, y]).T

split = ms.train_test_split(data, train_size = 0.8)

trainX1 = split[0].T[0]
trainX2 = split[0].T[1]
trainY = split[0].T[2]
testX1 = split[1].T[0]
testX2 = split[1].T[1]
testY = split[1].T[2]

trainX = np.array([np.ones(len(trainX1)), trainX1, trainX2]).T
testX = np.array([np.ones(len(testX1)), testX1, testX2]).T

p0 = 0.5

#for i in range(len(trainY)):
#    if trainY[i] == 0:
#        plt.scatter(trainX1[i], trainX2[i], c = "red")
#    elif trainY[i] == 1:
#        plt.scatter(trainX1[i], trainX2[i], c = "green")

def Sigmoid(lennardStinkt):
   return  1 / (1 + np.power(np.e, -lennardStinkt))

def LogLoss(theta, x, y):
    penis = 0
    for i in range(len(y)):
        penis += y[i] * np.log(Sigmoid(np.dot(theta.T, x[i]))) + (1 - y[i]) * np.log(1 - Sigmoid(np.dot(theta.T, x[i])))
    return -penis / len(y)
        
def GradLogLoss(theta, x, y):
    grad = 0
    for i in range(len(y)):
        grad += (Sigmoid(np.dot(theta.T, x[i])) - y[i]) * x[i]
    return grad / len(y)

def LogGradDesc(theta0, learnRate, n, step , x, y):
    theta = theta0
    loss = []
    for i in range(n):
        theta = theta - learnRate * GradLogLoss(theta, x, y)
#        if i % step == 0:
#            loss.append(LogLoss(theta, x, y))
#    plt.plot(np.arange(0, n, step), loss)
    return theta

def Predict(theta, x, p0 = 0.5):
    p = Sigmoid(np.dot(theta.T, x))
    if (p > p0):
        return 1
    else:
        return 0
    
def Results(theta, x, y, p0 = 0.5):
    tn = 0
    fn = 0
    fp = 0
    tp = 0
    for i in range(len(y)):
        if Predict(theta, x[i], p0) == 0:
            if y[i] == 0:
                tn += 1
            else:
                fn += 1
        else:
            if y[i] == 0:
                fp += 1
            else:
                tp += 1
    return [tp / (tp + fp), tp / (tp + fn), tp / (tp + (fn + fp) / 2), fp / (tn + fp)]

theta = LogGradDesc(np.array([0, 0, 0]), 0.001, 15000, 500, trainX, trainY)

p0 = np.arange(0.01, 0.98, 0.01)

precision = []
recall = []
f1 = []
fpr = []
for i in range(len(p0)):
    results = Results(theta, trainX, trainY, p0[i])
    precision.append(results[0])
    recall.append(results[1])
    f1.append(results[2])
    fpr.append(results[3])
    
plt.plot(recall, precision)
plt.plot(p0, f1)
plt.plot(fpr, recall)

precision = []
recall = []
f1 = []
fpr = []
for i in range(len(p0)):
    results = Results(theta, testX, testY, p0[i])
    precision.append(results[0])
    recall.append(results[1])
    f1.append(results[2])
    fpr.append(results[3])
    
plt.figure()
plt.plot(recall, precision)
plt.plot(p0, f1)
plt.plot(fpr, recall)

#fygge = plt.figure()
#ax = fygge.add_subplot(111, projection = '3d')
#for i in range(len(testY)):
#    ax.scatter(trainX1[i], trainX2[i], Sigmoid(np.dot(theta.T, trainX[i])), c = "blue")
#ax.plot([-3, 3], [-3, 3], [0.5, 0.5], c = "orange")