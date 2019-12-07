# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:38:15 2019

@author: mifiamigahna
"""

import numpy
import matplotlib.pyplot as pyplot

amount = 1000
middle = [0.5, 0.5]

distances = []
sample = numpy.random.rand(amount, 2)

j = 0
for i in sample:
    distances.append(numpy.sqrt(numpy.square(sample[j][0] - 0.5) + numpy.square(sample[j][1] - 0.5)))
    j += 1
    
histData = numpy.histogram(distances)

j = 0
histogramX = []
for i in histData[0]:
    histogramX.append((histData[1][j] + histData[1][j + 1]) / 2)
    j += 1
    
pyplot.subplot(131)
pyplot.plot(histogramX, histData[0])
    

normal = numpy.random.randn(amount)
    
histDataN = numpy.histogram(normal)

j = 0
histogramNX = []
for i in histDataN[0]:
    histogramNX.append((histDataN[1][j] + histDataN[1][j + 1]) / 2)
    j += 1

pyplot.subplot(132)
pyplot.plot(histogramNX, histDataN[0])
    

distances.sort()
normal.sort()

accuracy = 25
j = 1
ascend = []
for i in range(accuracy - 1):
    ascend.append(j / accuracy)
    j += 1
    
sampleQ = numpy.quantile(distances, ascend)
normalQ = numpy.quantile(normal, ascend)

pyplot.subplot(133)
pyplot.scatter(normalQ, sampleQ)
pyplot.plot([normalQ[0], normalQ[-1]], [sampleQ[0], sampleQ[-1]])