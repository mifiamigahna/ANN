# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 21:54:45 2019

@author: mifiamigahna
"""

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

def ann(x, w, b, v):
    y = 0
    for i in range(len(v)):
        y += v[i] * sp.expit(w * x + b[i])
    return y

def bias(w, s):
    return -w * s

x = np.arange(0, 1, 0.001)
w = 10000
s = [0.2, 0.4, 0.4, 0.6, 0.6, 0.8]
b = [bias(w, step) for step in s]
v = [0.5, -0.5, 0.8, -0.8, 0.2, -0.2]

yHut = []
for i in range(len(x)):
    yHut.append(ann(x[i], w, b, v))

plt.plot(x, yHut)


#3

n = 10
s = [x for x in [(x, -x)]]
meanS = [x for i, x in enumerate(np.arange(0, 1, (1 / (2 * n)))) if i % 2 != 0]
