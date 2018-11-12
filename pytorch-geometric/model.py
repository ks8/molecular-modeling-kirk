from argparse import Namespace

import torch.nn as nn

from chemprop.models.mpn import MPNEncoder
from chemprop.nn_utils import get_activation_function, initialize_weights, MayrDropout, MayrLinear


class MoleculeModel(nn.Module):
    def __init__(self, encoder: nn.Module, ffn: nn.Sequential):
        super(MoleculeModel, self).__init__()
        self.encoder = encoder
        self.ffn = ffn

    def forward(self, *input):
        return self.ffn(self.encoder(*input))


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a message passing neural network including final linear layers and initializes parameters.

    :param args: Arguments.
    :return: An nn.Module containing the MPN encoder along with final linear layers with parameters initialized.
    """
    # Regression with binning
    output_size = args.num_tasks

    encoder = MPNEncoder(args, args.atom_fdim, args.bond_fdim)

    first_linear_dim = args.hidden_size * (1 + args.jtnn)

    drop_layer = lambda p: nn.Dropout(p)
    linear_layer = lambda input_dim, output_dim, p: nn.Linear(input_dim, output_dim)

    # Create FFN layers
    if args.ffn_num_layers == 1:
        ffn = [
            drop_layer(args.ffn_input_dropout),
            linear_layer(first_linear_dim, output_size, args.ffn_input_dropout)
        ]
    else:
        ffn = [
            drop_layer(args.ffn_input_dropout),
            linear_layer(first_linear_dim, args.ffn_hidden_size, args.ffn_input_dropout)
        ]
        for _ in range(args.ffn_num_layers - 2):
            ffn.extend([
                get_activation_function(args.activation),
                drop_layer(args.ffn_dropout),
                linear_layer(args.ffn_hidden_size, args.ffn_hidden_size, args.ffn_dropout),
            ])
        ffn.extend([
            get_activation_function(args.activation),
            drop_layer(args.ffn_dropout),
            linear_layer(args.ffn_hidden_size, output_size, args.ffn_dropout),
        ])

    # Classification
    if args.dataset_type == 'classification':
        ffn.append(nn.Sigmoid())

    # Combined model
    ffn = nn.Sequential(*ffn)
    model = MoleculeModel(encoder, ffn)

    initialize_weights(model)

    return model
