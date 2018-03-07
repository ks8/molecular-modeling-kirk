""" File for plotting results of machine learning glasses project """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys, argparse
from math import *
import numpy as np
import time
import copy, os
import json
import random
import csv
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import linear_model

# Read in the cooling.dat data
cooling_dat = pd.read_csv('av_cooling.dat', sep='\t', error_bad_lines=False)

# Create an empty data numpy array 
data = np.zeros([cooling_dat.shape[0], 2])

# Fill in the data array
for i in range(cooling_dat.shape[0]):
	data[i, 0] = float(cooling_dat.iloc[i, 0].split()[0])
	data[i, 1] = float(cooling_dat.iloc[i, 0].split()[1])

# https://matplotlib.org/examples/axes_grid/demo_parasite_axes2.html
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA 

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()

host.plot(data[:, 0], data[:, 1])
#par1.plot([1.961, 1.56125, 1.415, 1.22, 1.025, 0.83, 0.635, 0.44], [99.833, 97.5466, 96.0663, 94.408, 88.8201, 75.9367, 62.3961, 48.7964], 'ro')
par1.plot([2.0, 1.9025, 1.85375, 1.75625, 0.635], [99.7, 99.9333, 99.9333, 99.6667, 96.3194], 'go')
host.set_xlabel('Temperature')
host.set_ylabel('Energy')
par1.set_ylabel('Test Accuracy')






plt.axvline(x=0.37)



# Regression on glassy end
# glassy = data[180:]

# # I want the data to be 19 x 1 lol...
# glassy_X = np.zeros([19, 1])
# for i in range(19):
# 	glassy_X[i, 0] = glassy[i, 0]


# glassy_y = np.zeros([19, 1])
# for i in range(19):
# 	glassy_y[i, 0] = glassy[i, 1]



# glassy_X_train = glassy_X
# glassy_y_train = glassy_y
# regr = linear_model.LinearRegression()
# regr.fit(glassy_X_train, glassy_y_train)

# fulldata = np.zeros([199, 1])
# for i in range(199):
# 	fulldata[i, 0] = data[i, 0]

# print(regr.predict(fulldata))






plt.plot()
plt.title('Energy and ML Accuracy')
plt.savefig('accuracies.png')




# plt.plot(data[:, 0], data[:, 1])
# plt.axvline(x=0.21)
# plt.plot()
# plt.savefig('test.png')







def indices_containing_substring(the_list, substring):
	indices = []
	for i, s in enumerate(the_list):
		if substring in s:
			indices.append(i)	
	return indices



















