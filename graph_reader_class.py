"""File to createa a dataset class that is capable to construct an adjacency matrix, feature list, distance dictionary, and target list for one specific timestep of one specific simulation trajectory, by Kirk Swanson"""

# Load python modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys, argparse
from math import *
import numpy as np
import time
import copy, os
from ast import literal_eval as make_tuple
import json
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import random
import math
import operator
import csv
import pandas as pd 
from sklearn.metrics.pairwise import euclidean_distances


class GlassyData():

	# Constructor
	def __init__(self, root_path, metadata, num_neighbors, e_representation = 'raw_distance'):

		self.root = root_path
		self.metadata = metadata
		self.num_neighbors = num_neighbors
		self.e_representation = e_representation

	# Get item
	def __getitem__(self, index):

		# Load data
		data = np.loadtxt(os.path.join(self.root, self.metadata[index]))

		# Compute all pairwise distances
		distances = euclidean_distances(data[:, 1:3], data[:, 1:3])

		# Create an empty adjacency matrix
		adj_matrix = np.zeros((4320, 4320))	

		# Dictionary to hold all graph connected pairwise distances
		graph_distances = dict()

		# Compute the adjacency matrix and dictionary of edge distances
		for i in range(data.shape[0]):
			nearest = distances[i].argsort()[1:self.num_neighbors+1]
			adj_matrix[i, :][np.array(nearest)] = 1
			adj_matrix[:, i][np.array(nearest)] = 1
			for k in range(self.num_neighbors):
				if not (((i, nearest[k]) in graph_distances) or ((nearest[k], i) in graph_distances)):
					graph_distances[(min(i, nearest[k]), max(i, nearest[k]))] = [distances[i, nearest[k]], 1, 0, 0, 0]

		# Check that adjacency matrix corresponds to a connected graph
	
		# List to hold atom features
		features = list()
	
		# Compute the target property
		if 'glass' in self.metadata[index]:
			target = np.asarray([1.0, 0.0])
		else:
			target = np.asarray([0.0, 1.0])

		# Compute the atom features
		for i in range(data.shape[0]):
			features.append([data[i][0]])
	

		# Return adjacency matrix, feature list, distances dictionary, and target list
		return (adj_matrix, features, graph_distances), target

	# Length computation
	def __len__(self):

		return len(self.metadata)

				

