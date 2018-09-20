"""Construct a nearest neighbor graph from node position data, adapted from PyTorch Geometric repository at https://github.com/rusty1s/pytorch_geometric, by Kirk Swanson"""
# Load modules
import torch
import scipy.spatial
from undirected_PyTorch import to_undirected

class NNGraph(object):
	"""Class for transformation of node position data to nearest neighbor graph.  INitialize via nngraph = NNGraph(k), call via data = nngraph(data)"""

	def __init__(self, k=5):
		self.k = k

	def __call__(self, data):
		pos = data.pos
		assert not pos.is_cuda

		