# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:30:39 2019

@author: mifiamigahna
"""

import numpy as np
import matplotlib.pyplot as plt

#1

x = np.arange(-50, 51)
    
plt.subplot(121)
plt.plot(x, np.square(x))
plt.subplot(122)
plt.plot(x, np.sqrt(x))
plt.plot(x, -np.sqrt(x))


#2

x = np.arange(-10, 11)

def FuncA(x):
    return 2 * x + 2

def FuncB(x):
    return 5 * np.square(x) + x

def FuncC(x):
    return 11 * np.power(x, 3) + 2 * np.square(x) + 2 * x + 3

def FuncD(x):
    return np.exp(x)

plt.figure
plt.subplot(2,2,1)
plt.plot(x, FuncA(x))
plt.subplot(2,2,2)
plt.plot(x, FuncB(x))
plt.subplot(2,2,3)
plt.plot(x, FuncC(x))
plt.subplot(2,2,4)
plt.plot(x, FuncD(x))


m = np.arange(-1, 2)
b = np.arange(-5, 10, 5)
x = np.arange(-10, 11)

plt.figure
plt.subplot(331)
plt.plot(x, m[0] * x + b[0])
plt.subplot(332)
plt.plot(x, m[1] * x + b[0])
plt.subplot(333)
plt.plot(x, m[2] * x + b[0])
plt.subplot(334)
plt.plot(x, m[0] * x + b[1])
plt.subplot(335)
plt.plot(x, m[1] * x + b[1])
plt.subplot(336)
plt.plot(x, m[2] * x + b[1])
plt.subplot(337)
plt.plot(x, m[0] * x + b[2])
plt.subplot(338)
plt.plot(x, m[1] * x + b[2])
plt.subplot(339)
plt.plot(x, m[2] * x + b[2])


