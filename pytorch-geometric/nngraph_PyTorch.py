"""Construct a nearest neighbor graph from node position data, adapted from PyTorch Geometric repository at https://github.com/rusty1s/pytorch_geometric, by Kirk Swanson"""
# Load modules
from __future__ import print_function, division
import torch
import scipy.spatial
from undirected_PyTorch import to_undirected

class NNGraph(object):
	"""Class for transformation of node position data to nearest neighbor graph.  Initialize via nngraph = NNGraph(k), call via data = nngraph(data)"""

	def __init__(self, k=5):
		"""
		Args:
			k (int): Number of nearest neighbors to compute for each node  
		"""
		self.k = k

	def __call__(self, data):
		pos = data.pos
		assert not pos.is_cuda

		row = torch.arange(pos.size(0), dtype=torch.long)
		row = row.view(-1, 1).repeat(1, self.k).view(-1)

		_, col = scipy.spatial.cKDTree(pos).query(pos, self.k + 1)
		col = torch.tensor(col)[:, 1:].contiguous().view(-1)
		mask = col < pos.size(0)
		edge_index = torch.stack([row[mask], col[mask]], dim=0)
		edge_index = to_undirected(edge_index, num_nodes=pos.size(0))

		data.edge_index = edge_index
		return data

	def __repr__(self):
		return '{}(k={})'.format(self.__class__.__name__, self.k)




 