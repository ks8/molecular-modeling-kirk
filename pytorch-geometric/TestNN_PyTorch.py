"""Test neural network for glass data, by Kirk Swanson"""
# Load modules
from GlassDataset_PyTorch import GlassDataset
from nngraph_PyTorch import NNGraph 
from distance_PyTorch import Distance 
from compose_PyTorch import Compose
from dataloader_PyTorch import DataLoader
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

# Load the dataset from metadata folder, which is constructed using GlassMetadata.py
dataset = GlassDataset('metadata/metadata.json', transform=Compose([NNGraph(5), Distance(False)]))

# Create a dataloader
dataloader = DataLoader(dataset, batch_size=2)

dataloader = iter(dataloader)

example = dataloader.next()

print(example.x.size())
print(example.x)


exit()


# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()

#         self.fc1 = nn.Linear(1, 28)
#         self.fc2 = nn.Linear(28, 32)
#         self.fc3 = nn.Linear(32, 2)


#     def forward(self, x):
#     	x = F.relu(self.fc1(x))
#     	x = F.relu(self.fc2(x))
#     	x = F.relu(self.fc3(x))

#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square you can only specify a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features


# net = Net()
# print(net)









