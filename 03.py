# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:52:01 2019

@author: mifiamigahna
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics as met
from sklearn import linear_model as lin
from random import shuffle

def Value(x, coef, e = []):
    if type(x) is list:
        if type(x[0]):
            y = []
            for i in range(len(x)):
                if len(e) == 0:
                    y.append(0)
                else:
                    y.append(e[i])
                for j in range(len(x[i])):
                    y[i] += coef[j] * x[i][j]
            return y
        else:
            y = []
            for i in range(len(x)):
                y.append(0)
                for j in range(len(x[i])):
                    y[i] += coef[j] * x[i][j]
            return y
    else:
        y = 0
        i = 0
        for j in coef:
            y += j * np.power(x, i)
            i += 1
        return y

def Derivative(coef):
    coefNew = []
    for i in range(len(coef)):
        if i > 0:
            coefNew.append(coef[i] * (i))
    return coefNew

def GradDesc(coef, learnRate, n , x0):
    coef1 = Derivative(coef)
    for i in range(n):
        x0 -= learnRate * Value(x0, coef1)
    return x0

#1
    
def Cov(x = [], y = []):
    sum1 = 0
    n = len(x)
    for i in range(n):
        sum1 += x[i] * y[i]
    return sum1 / n - (np.sum(x) * np.sum(y)) / np.square(n)

def Var(x = [], y = []):
    sum1 = 0
    n = len(x)
    for i in range(n):
        sum1 += np.square(x[i] - np.mean(x))
    return sum1 / (n - 1)
    

coef = [5, 4]
x = np.linspace(-3, 3, 100)
e = np.random.uniform(-0.5, 0.5, 100)
y = []

for i in range(len(x)):
    y.append(Value(x[i], coef) + e[i])

#plt.scatter(x, y)

m = Cov(x, y) / Var(x)
b = np.mean(y) - m * np.mean(x)

#print("m = " + str(m))
#print("b = " + str(b))

def Transpose(x):
    t = []
    if type(x[0]) is int:
        for i in range(len(x)):
            t.append([x[i]])
    else:
        for i in range(len(x[0])):
            t.append([])
        for j in range(len(x)):
            for i in range(len(x[j])):
                t[i].append(x[j][i])
    return t
    

def VecMult(x, y):
    if type(y) is not list:
        v = []
        for i in range(len(x)):
            v.append(x[i] * y)
        return v
    elif len(x) == len(y):
        v = 0
        for i in range(len(x)):
            v += x[i] * y[i]
        return v
    else:
        print("Banana!")

def MatMult(x, y):
    if len(x[0]) == len(y):    
        m = []
        if type(y[0]) is not list:
            for i in range(len(x)):
                m.append(0)
                for j in (range(len(x[i]))):
                    m[i] += x[i][j] * y[j]
        else:    
            yT = Transpose(y)
            for i in range(len(x)):
                m.append([])
                for j in range(len(yT)):
                    m[i].append(VecMult(x[i], yT[j]))
        return m
    else:
        print("use your imagination")

def SubMatrix(x, i, j):
    uD = []
    m = 0
    for k in range(len(x)):
        if k != i:
            uD.append([])
            for l in range(len(x[k])):
                if l != j:
                    uD[m].append(x[k][l])
            m += 1
    return uD
    
def Determination(x):
    if len(x) == len(x[0]):
        d = 0
        if len(x) == 1:
            d = x[0][0]
        elif len(x) == 2:
            d = x[0][0] * x[1][1] - x[0][1] * x[1][0]
        else:
            for i in range(len(x)):
                d += x[0][i] * np.power(-1, i) * Determination(SubMatrix(x, 0, i))
                
        return d
    else:
        print("do you wanna have a bad tom?")

def Cof(x, i, j):
    return np.power(-1, i + j) * Determination(SubMatrix(x, i, j))

def CofMatrix(x):
    cM = []
    for i in range(len(x)):
        cM.append([])
        for j in range(len(x[0])):
            cM[i].append(Cof(x, i, j))
    return cM

def ScalMatMult(x, s):
    m = []
    for i in range(len(x)):
        m.append([])
        for j in range(len(x[i])):
            m[i].append(x[i][j] * s)
    return m

def Inverse(x):
    d = Determination(x)
    if d != 0:
        return  ScalMatMult(Transpose(CofMatrix(x)), 1 / d)

def Sample(rMin, rMax, n, d, p = "SchwanZ"):
    x = []
    if type(p) is not str:
        l = np.linspace(rMin, rMax, n)
        for i in range(n):
            x.append([])
            for j in range(len(p)):
                x[i].append(np.power(l[i], p[j]))
    else:
        for i in range(n):
            x.append([1])
            for j in range(d - 1):
                x[i].append(np.random.uniform(rMin,rMax))
    return x
    

def ThetaHat(x, y):
    return MatMult(MatMult(Inverse(MatMult(Transpose(x), x)), Transpose(x)), y)

def F(x1, x2, coef):
    return coef[0] + coef[1] * x1 + coef[2] * x2

def Mean(x):
    m = 0
    for i in x:
        m += i
    return m / len(x)

def SS(x, m):
    s = 0
    for i in range(len(x)):
        s += np.square(x[i] - m)
    return s

def Saldo(x, y):
    s = []
    for i in range(len(x)):
        s.append(x[i] - y[i])
    return s

coef = [3, 7, -4]
x = Sample(-3, 3, 100, 3)
e = np.random.uniform(-0.5, 0.5, 100)
y = Value(x, coef, e)
coefHat = ThetaHat(x, y)
yHat = Value(x, coefHat)
r2 = SS(yHat, Mean(y)) / SS(y, Mean(y))

print(MatMult(MatMult(Inverse(MatMult(Transpose(x), x)), Transpose(x)), y))
print(r2)
figgis = plt.figure()
ax = figgis.add_subplot(111, projection = '3d')
ax.scatter(Transpose(x)[1], Transpose(x)[2], y)
ax.plot([-3, 3], [-3, 3], [F(-3, -3, coefHat), F(3, 3, coefHat)])
plt.figure()
axes = plt.axes()
axes.set_ylim([-10, 10])
axes.set_xlim([-36, 36])
plt.scatter(y, Saldo(y, yHat))


#e

def Polynom(x, coef, e):
    y = []
    for i in range(len(x)):
        y.append(0)
        for j in range(len(coef)):
            y[i] += coef[j] * np.power(x[i], j) +  e[i]
    return y

coef = [5, 4]
x = np.array(np.linspace(-3, 3, 100)).reshape(-1,1)
y = Polynom(x, coef, e)
reg = lin.LinearRegression().fit(x, y)
yHat = reg.predict(x)

plt.figure()
plt.scatter(x, y)

print(reg.coef_)
print(reg.intercept_)
print(met.r2_score(y, yHat))


#2

def L2Norm(y, yHat):
    l2 = 0
    for i in range(len(y)):
        l2 += np.sqrt(np.square(y[i] - yHat[i]))
    return l2

def L2Plot(n, step, coef):
    plortX = []
    plortY = []
    for i in range(int(n / step)):
        if (step * i) > len(coef):
            x = Sample(-2, 2, step * i + 1, "Penis", p = np.arange(6))
            e = np.random.uniform(-1, 1, 100)
            y = Value(x, coef, e)
            coefHat = ThetaHat(x, y)
            yHat = Value(x, coefHat)
            l2 = L2Norm(y, yHat)
            plortX.append(step * i)
            plortY.append(l2)
    plt.scatter(plortX, plortY)

coef = [3, 4, -2, 1.5, 0, -1]
x = Sample(-2, 2, 100, "Penis", p = np.arange(6))
e = np.random.uniform(-1, 1, 100)
y = Value(x, coef, e)
coefHat = ThetaHat(x, y)
yHat = Value(x, coefHat)
r2 = SS(yHat, Mean(y)) / SS(y, Mean(y))
l2 = L2Norm(y, yHat)

print(MatMult(MatMult(Inverse(MatMult(Transpose(x), x)), Transpose(x)), y))
print(r2)
print(l2)
plt.figure()
plt.scatter(Transpose(x)[1], y)
plt.figure()
axes = plt.axes()
axes.set_ylim([-10, 10])
axes.set_xlim([-20, 20])
plt.scatter(y, Saldo(y, yHat))

plt.figure()
L2Plot(100, 5, coef)


#3

def ListSum(a, b):
    s = []
    for i in range(len(a)):
        s.append(a[i] + b[i])
    return s

def UpdateGradDesc(guess, learnRate, y, x, iterations):
    theta = guess
    for i in range(iterations):
        c = list(zip(x, y))
        shuffle(c)
        x, y = zip(*c)
        for j in range(len(x)):
            theta = ListSum(theta, VecMult(x[j], learnRate* (y[j] - VecMult(x[j], theta))))
    return theta

def EuclidPlot(n, step, coef):
    plortX = []
    plortY = []
    for i in range(int(n / step)):
        if (step * i) > len(coef):
            x = Sample(-3, 3, step * i + 1, 2)
            e = np.random.uniform(-0.5, 0.5, 100)
            y = Value(x, coef, e)
            coefHat = UpdateGradDesc([0, 0], 0.01, y, x, 10)
            yHat = Value(x, coefHat)
            l2 = L2Norm(y, yHat)
            plortX.append(step * i)
            plortY.append(l2)
    plt.scatter(plortX, plortY)

coef = [5, 4]
x = Sample(-3, 3, 100, 2)
e = np.random.uniform(-0.5, 0.5, 100)
y = Value(x, coef, e)

thetaHat = UpdateGradDesc([0, 0], 0.01, y, x, 10)
print(thetaHat)
plt.figure()
EuclidPlot(100, 5, coef)