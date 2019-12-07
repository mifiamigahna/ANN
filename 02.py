# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:01:26 2019

@author: mifiamigahna
"""

import numpy as np
import matplotlib.pyplot as plt

#1

x = np.arange(-2, 3, 0.1)
def Func(x):
    return np.power(x, 4) - 4 * np.square(x) + 4

plt.plot(x, Func(x))
plt.scatter(0, 4, c = "green")
plt.scatter([- np.sqrt(2), np.sqrt(2)], [0, 0], c = "red")


#2

coef = [4, 0, -4, 0, 1]

def Value(x, coef):
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

descX = []
descY = []
xTemp = 0

for i in x:
    xTemp = GradDesc(coef, 0.01, 50, i)
    descX.append(xTemp)
    descY.append(Value(xTemp, coef))

plt.scatter(descX, descY, c = "yellow")