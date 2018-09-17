"""Custom PyTorch dataset class for loading initial 2D glass data, by Kirk Swanson"""
# Load modules 
from __future__ import print_function, division
import json
import os
import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, utils 

# Dataset class
class GlassyDataset(Dataset):
	""" Custom dataset for 2D Glassy data"""

	def __init__(self, metadata_file, root_dir, transform=None):
		"""
		Args:
			metadata (string): Path to the metadata file
			root (string): Directory with all the coordinate files 
		"""
		super(Dataset, self).__init__()
		self.metadata = json.load(open(metadata_file, 'r'))
		self.root = root_dir
		self.transform=transform

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, idx):
		coords_file = np.loadtxt(self.metadata[idx]['path'])

		if self.transform:
			coords_file = self.transform(coords_file)

		return coords_file