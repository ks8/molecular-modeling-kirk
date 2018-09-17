import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from GlassDataset_PyTorchGeometric import GlassyDataset


num_neighbors = 6
dataset = GlassyDataset('metadata/metadata.json', transform=T.NNGraph(num_neighbors))

print(dataset[0])
print(dataset[0].x)
print(dataset[0].edge_index)
print(dataset[0].pos)
print(dataset[0].y)




