"""Test neural network for glass data, by Kirk Swanson (and Kyle Swanson)"""

from typing import List, Tuple

import torch

from GlassDataset_PyTorch import GlassDataset
from nngraph_PyTorch import NNGraph 
from distance_PyTorch import Distance 
from compose_PyTorch import Compose
from dataloader_PyTorch import DataLoader

# Load the dataset from metadata folder, which is constructed using GlassMetadata.py
# TODO: From Shubhendu, make number of neighbors log(n)
dataset = GlassDataset('metadata/metadata.json', transform=Compose([NNGraph(5), Distance(False)]))

# Create a dataloader
batch_size = 5
dataloader = DataLoader(dataset, batch_size=batch_size)
dataloader = iter(dataloader)


class GlassBatchMolGraph:
    def __init__(self, example):
        self.atom_fdim = example.pos.size(1) + example.x.size(1)
        bond_fdim = example.edge_attr.size(1)
        self.bond_fdim = self.atom_fdim + bond_fdim  # bond features are really combined atom/bond features

        self.n_atoms = 1  # number of atoms (+1 for padding)
        self.n_bonds = 1  # number of bonds (+1 for padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        self.f_atoms = [torch.zeros(self.atom_fdim)]  # padding
        self.f_bonds = [torch.zeros(self.bond_fdim)]  # padding
        self.a2b = [[]]
        self.b2a = [0]
        self.b2revb = [0]

        # atoms
        count = 0
        current_example_index = 0
        for example_index, pos, x in zip(example.batch, example.pos, example.x):
            self.f_atoms.append(torch.cat((pos, x), dim=0))

            if example_index != current_example_index:
                start = self.n_atoms - count
                self.a_scope.append((start, count))
                current_example_index = example_index
                count = 0

            count += 1
            self.n_atoms += 1
            self.a2b.append([])

        start = self.n_atoms - count
        self.a_scope.append((start, count))

        # bonds
        count = 0
        current_example_index = 0
        for (a1, a2), edge_attr in zip(example.edge_index.t(), example.edge_attr):
            a1, a2 = a1.item(), a2.item()
            example_index = example.batch[a1]

            a1, a2 = a1 + 1, a2 + 1  # +1 for padding

            self.f_bonds.append(torch.cat((self.f_atoms[a1], edge_attr), dim=0))
            self.f_bonds.append(torch.cat((self.f_atoms[a2], edge_attr), dim=0))

            b1 = self.n_bonds  # b1 = a1 --> a2
            b2 = b1 + 1  # b2 = a2 --> a1
            self.a2b[a2].append(b1)
            self.b2a.append(a1)
            self.a2b[a2].append(b2)
            self.b2a.append(a2)
            self.b2revb.append(b2)
            self.b2revb.append(b1)

            if example_index != current_example_index:
                start = self.n_bonds - count
                self.b_scope.append((start, count))
                current_example_index = example_index
                count = 0

            count += 2  # 2 b/c directed edges
            self.n_bonds += 2

        start = self.n_bonds - count
        self.b_scope.append((start, count))

        # Cast to tensor
        self.max_num_bonds = max(len(in_bonds) for in_bonds in self.a2b)
        self.f_atoms = torch.stack(self.f_atoms)
        self.f_bonds = torch.stack(self.f_bonds)
        self.a2b = torch.LongTensor([self.a2b[a] + [0] * (self.max_num_bonds - len(self.a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(self.b2a)
        self.b2revb = torch.LongTensor(self.b2revb)

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope


example = next(dataloader)
targets = example.y
graph = GlassBatchMolGraph(example)

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









