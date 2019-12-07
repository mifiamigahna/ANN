# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:13:46 2019

@author: mifiamigahna
"""

import numpy as np
import sklearn as skl
import sklearn.model_selection as ms
import sklearn.linear_model as lin
import matplotlib.pyplot as plt

deg = [1, 2, 3, 4, 5, 6]
k = 20

data = np.load("C:\\Users\\mifiamigahna\\documents\\04_model_selection_data.npy")
split = ms.train_test_split(data, train_size = 0.8)

trainX = split[0].T[0].reshape(-1, 1)
trainY = split[0].T[1].reshape(-1, 1)
testX = split[1].T[0].reshape(-1, 1)
testY = split[1].T[1].reshape(-1, 1)

def meanSqrErr(y, yHat):
    e = 0
    for i in range(len(y)):
        e += np.square(y[i] - yHat[i])
    return float(e / len(y))

def polyFit(trainX, trainY, testX, testY, degree):
    mse1 = []
    mse2 = []
    #plt.figure()
    for i in degree:
        poly = skl.preprocessing.PolynomialFeatures(i)
        features = poly.fit_transform(trainX)
        
        reg = lin.LinearRegression()
        f = reg.fit(features, trainY)
        
        x = np.arange(-1, 1, 0.01).reshape(-1, 1)
        yHat = np.polyval(np.flip(f.coef_[0]), x)
        
        trainYHat = np.polyval(np.flip(f.coef_[0]), trainX)
        testYHat = np.polyval(np.flip(f.coef_[0]), testX)
        
        mse1.append(meanSqrErr(trainY, trainYHat))
        mse2.append(meanSqrErr(testY, testYHat))
        
#        axes = plt.axes()
#        axes.set_ylim([-1.5, 3])
#        axes.plot(x, yHat)
#        axes.scatter(trainX, trainY)
        
#    plt.figure()
#    plt.scatter(degree, mse1)
#    plt.scatter(degree, mse2)
    plt.scatter(0.0035, mse1)
    plt.scatter(0.0035, mse2)
        
    return mse1, mse2
        
#polyFit(trainX, trainY, testX, testY, deg)


#2

def kFold(data, k, deg):
    fold = ms.KFold(k, True)
    mmse1 = []
    mmse2 = []
    mmmses = []

    for i, j in fold.split(data):
        trainX = data[i].T[0].reshape(-1, 1)
        trainY = data[i].T[1].reshape(-1, 1)
        testX = data[j].T[0].reshape(-1, 1)
        testY = data[j].T[1].reshape(-1, 1)
        
        mses = polyFit(trainX, trainY, testX, testY, deg)
        mmse1.append(mses[0])
        mmse2.append(mses[1])
        
    mmse1 = list(np.array(mmse1).T)
    mmse2 = list(np.array(mmse2).T)
        
    for i in range(len(mmse1)):
        mmse1[i] = np.mean(mmse1[i])
        mmse2[i] = np.mean(mmse2[i])
        mmmses.append((mmse1[i] + mmse2[i]) / 2)
    
#    plt.figure()
#    plt.scatter(range(1, deg + 1), mmse1)
#    plt.scatter(range(1, deg + 1), mmse2)
#    plt.scatter(range(1, deg + 1), mmmses)
#    print("Best:", np.argmin(mmmses) + 1)
         
    return np.argmin(mmmses) + 1
    
#print(kFold(data, k, deg))
    

#3
    
def ridgeReg(trainX, trainY, testX, testY, deg):
    mse1 = []
    mse2 = []
    
    poly = skl.preprocessing.PolynomialFeatures(deg)
    features = poly.fit_transform(trainX)
    
    for i in np.arange(0, 0.003, 0.0005):
        reg = lin.Ridge(i)
        f = reg.fit(features, trainY)
        
        x = np.arange(-1, 1, 0.01).reshape(-1, 1)
        yHat = np.polyval(np.flip(f.coef_[0]), x)
        
        trainYHat = np.polyval(np.flip(f.coef_[0]), trainX)
        testYHat = np.polyval(np.flip(f.coef_[0]), testX)
        
        mse1.append(meanSqrErr(trainY, trainYHat))
        mse2.append(meanSqrErr(testY, testYHat))
    
#    plt.plot(x, yHat)
#    plt.scatter(trainX, trainY)
#    plt.figure()
    axes = plt.axes()
    axes.set_ylim([0, 1])
    axes.scatter(np.arange(0, 0.003, 0.0005), mse1)
    axes.scatter(np.arange(0, 0.003, 0.0005), mse2)
    
    plt.figure()
    
    
ridgeReg(trainX, trainY, testX, testY, 9)
polyFit(trainX, trainY, testX, testY, [3])