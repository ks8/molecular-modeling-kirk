"""Test neural network for glass data, by Kirk Swanson"""
# Load modules
from GlassDataset_PyTorch import GlassDataset
from nngraph_PyTorch import NNGraph 
from distance_PyTorch import Distance 
from compose_PyTorch import Compose
from dataloader_PyTorch import DataLoader 

# Load the dataset from metadata folder, which is constructed using GlassMetadata.py
dataset = GlassDataset('metadata/metadata.json', transform=Compose([NNGraph(5), Distance(False)]))

# Create a dataloader
dataloader = DataLoader(dataset, batch_size=2)



