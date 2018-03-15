"""File to construct an adjacency matrix, feature list, distance dictionary, and target list for one specific timestep of one specific simulation trajectory, by Kirk Swanson"""

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import math
import operator
import csv
import pandas as pd 
from sklearn.metrics.pairwise import euclidean_distances

# Function to parse arguments 
def create_parser():

	# Create parser and add arguments
	parser = argparse.ArgumentParser(description='Read data files')
	parser.add_argument('-datafile', dest='datafile', default=None, help='Name of data file')
	parser.add_argument('-num_neighbors', dest='num_neighbors', type=int, default=7, help='Number of nearest neighbors to consider when computing adjacency matrix')
	
	return parser

# Function to convert arguments into a dictionary
def convert_args(args):

	# Files dictionary
	files={}
	files['datafile'] = args.datafile
	
	options = {}
	options['num_neighbors'] = args.num_neighbors

	return files, options

def process_data(files, options):

	# Open and read the datafile	
	data = np.loadtxt(files['datafile'])

	# Load options
	num_neighbors = options['num_neighbors']

	# Compute all pairwise distances
	distances = euclidean_distances(data[:, 1:3], data[:, 1:3])

	# Create an empty adjacency matrix
	adj_matrix = np.zeros((4320, 4320))	

	# Dictionary to hold all graph connected pairwise distances
	graph_distances = dict()

	# Compute the adjacency matrix and dictionary of edge distances
	for i in range(data.shape[0]):
		nearest = distances[i].argsort()[1:num_neighbors+1]
		adj_matrix[i, :][np.array(nearest)] = 1
		adj_matrix[:, i][np.array(nearest)] = 1
		for k in range(num_neighbors):
			if not (((i, nearest[k]) in graph_distances) or ((nearest[k], i) in graph_distances)):
				graph_distances[(min(i, nearest[k]), max(i, nearest[k]))] = [distances[i, nearest[k]], 1, 0, 0, 0]

	# Check that adjacency matrix corresponds to a connected graph
	
	# List to hold atom features
	features = list()
	
	# Compute the target property
	if 'glass' in files['datafile']:
		target = np.asarray([1.0, 0.0])
	else:
		target = np.asarray([0.0, 1,0])

	# Compute the atom features
	for i in range(data.shape[0]):
		features.append([data[i][0]])

	# print(adj_matrix, features, graph_distances, target)

	# Return adjacency matrix, feature list, distances dictionary, and target list
	return adj_matrix, features, graph_distances, target


def main(argv):

	parser = create_parser()
	args = parser.parse_args()
	files, options = convert_args(args)

	process_data(files, options)

if __name__ == "__main__":
	main(sys.argv[1:])