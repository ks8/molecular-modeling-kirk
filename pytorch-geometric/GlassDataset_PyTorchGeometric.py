"Custom PyTorch dataset class for loading initial 2D glass data, by Kirk Swanson"
import sys
import json
import numpy as np
import os
import torch
from torch_geometric.data import Data, Dataset 

class GlassyDataset(Dataset):

	def __init__(self, metadata_file, transform=None):
		"""
		Args:
			metadata_file (string): Path to the metadata file
		"""
		super(Dataset, self).__init__()
		self.metadata = json.load(open(metadata_file, 'r'))
		self.transform=transform

	def raw_file_names(self):
		return None

	def processed_file_names(self):
		return None

	def download(self):
		return None

	def process(self):
		return None

	def __len__(self):
		return len(self.metadata)

	def get(self, idx):
		""" Output a data object with node features, positions, and target value"""
		coords_file = np.loadtxt(self.metadata[idx]['path'])
		data = Data()
		data.pos = torch.tensor(coords_file[:, 1:], dtype=torch.float)
		data.x = torch.tensor([[x] for x in coords_file[:, 0]], dtype=torch.float)
		if self.metadata[idx]['label'] == 'glass':
			data.y = torch.tensor([0])
		else:
			data.y = torch.tensor([1])

		return data




