"""File to construct an adjacency matrix for one specific timestep of one specific simulation trajectory, by Kirk Swanson"""

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

# Function to parse arguments 
def create_parser():

	# Create parser and add arguments
	parser = argparse.ArgumentParser(description='Read data files')
	parser.add_argument('-datafile', dest='datafile', default=None, help='Name of data file')
	parser.add_argument('-num_neighbors', dest='num_neighbors', default=7, help='Number of nearest neighbors to consider when computing adjacency matrix')
	
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
	data = open(files['datafile'], 'r')
	data_contents = data.readlines()
	data.close()

	# Load options
	num_neighbors = options['num_neighbors']

	# Dictionary to hold all pairwise distances 
	distances = dict()

	# Create an empty adjacency matrix
	adj_matrix = np.zeros((4320, 4320))	

	# Dictionary to hold all graph connected pairwise distances
	graph_distances = dict()

	# Compute all pairwise distances
	for i in range(len(data_contents)):
		print(i)
		# single_particle_distances = []
		for j in range(i, len(data_contents)):
			delta_x = float(data_contents[i].split()[1]) - float(data_contents[j].split()[1])
			delta_y = float(data_contents[i].split()[2]) - float(data_contents[j].split()[2])
			euclidean = math.sqrt(pow(delta_x, 2) + pow(delta_y, 2))
			distances[(i, j)] = euclidean
			distances[(j, i)] = euclidean

	# Compute the adjacency matrix and the dictionary of edge distances
	for i in range(len(data_contents)):
		print(i)
		particle_distances = {(i, j): distances[(i, j)] for j in range(4320)}
		particle_distances.pop((i, i), None)
		sorted_particle_distances = sorted(particle_distances.items(), key=operator.itemgetter(1))
		for k in range(num_neighbors):
			if not (((sorted_particle_distances[k][0][0], sorted_particle_distances[k][0][1]) in graph_distances) or (((sorted_particle_distances[k][0][1], sorted_particle_distances[k][0][0]) in graph_distances))):
				graph_distances[sorted_particle_distances[k][0]] = sorted_particle_distances[k][1]
				adj_matrix[sorted_particle_distances[k][0][0], sorted_particle_distances[k][0][1]] = 1
				adj_matrix[sorted_particle_distances[k][0][1], sorted_particle_distances[k][0][0]] = 1



def main(argv):

	parser = create_parser()
	args = parser.parse_args()
	files, options = convert_args(args)

	process_data(files, options)

if __name__ == "__main__":
	main(sys.argv[1:])