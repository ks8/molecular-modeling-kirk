""" File for plotting average of cooling.dat files in order to resolve the true average glass transition temperature for our simulation model """
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


# Upload the first cooling.dat file in order to create an empty data numpy array of the correct shape
cooling_dat = pd.read_csv('0/cooling.dat', sep='\t', error_bad_lines=False)
data = np.zeros([cooling_dat.shape[0], 2])
for i in range(cooling_dat.shape[0]):
	data[i, 0] = float(cooling_dat.iloc[i, 0].split()[0])
	data[i, 1] = float(cooling_dat.iloc[i, 0].split()[1])

# Now load and aggregate all of the cooling.dat data
for i in range(10000):
	print(i)
	cooling_dat = pd.read_csv(str(i) +'/' + 'cooling.dat', sep='\t', error_bad_lines=False)

	# Fill in the data array
	for j in range(cooling_dat.shape[0]):
		data[j, 1] += float(cooling_dat.iloc[j, 0].split()[1])

# Take the average of the energy values
for i in range(cooling_dat.shape[0]):
	data[i, 1]  = data[i, 1] / 10000.0

# Plot the data
plt.plot(data[:, 0], data[:, 1])
plt.savefig('average_cooling_data.png')

# Save the averaged data to a text file for future use
output = open('av_cooling.dat', 'w')
for i in range(data.shape[0]):
	output.write(str(data[i, 0]) + '    ' + str(data[i, 1]) + '\n')
output.close()




















