"""Custom PyTorch dataset class for loading initial 2D glass data, by Kirk Swanson"""
# Load modules 
from __future__ import print_function, division
import json
import os
import torch
import numpy as np 
from torch.utils.data import Dataset
from Data_PyTorch import Data 


# Dataset class
class GlassDataset(Dataset):
	""" Custom dataset for 2D glass data"""

	def __init__(self, metadata_file, transform=None):
		"""
		Args:
			metadata (string): Path to the metadata file
		"""
		super(Dataset, self).__init__()
		self.metadata = json.load(open(metadata_file, 'r'))
		self.transform=transform

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, idx):
		""" Output a data object with node features, positions, and target value, transformed as required"""
		coords_file = np.loadtxt(self.metadata[idx]['path'])
		data = Data()
		data.pos = torch.tensor(coords_file[:, 1:], dtype=torch.float)
		data.x = torch.tensor([[x] for x in coords_file[:, 0]], dtype=torch.float)
		if self.metadata[idx]['label'] == 'glass':
			data.y = torch.tensor([0])
		else:
			data.y = torch.tensor([1])

		data = data if self.transform is None else self.transform(data)

		return data

	def __repr__(self):
		return '{}({})'.format(self.__class__.__name__, len(self))



