from typing import List, Tuple

import torch


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
