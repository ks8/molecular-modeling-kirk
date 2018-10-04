"""Compose multiple transforms together, adapted from PyTorch Geometric repository at https://github.com/rusty1s/pytorch_geometric, by Kirk Swanson"""
# Load modules
from __future__ import print_function, division

class Compose(object):
    """Composes several transforms together.  Use as transform = Compose([NNGraph(5), Distance(False)])"""  

    def __init__(self, transforms):
        """
        Args:
            transforms (list of transform objects): List of transforms to compose.
        """
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        args = ['    {},'.format(t) for t in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, '\n'.join(args))