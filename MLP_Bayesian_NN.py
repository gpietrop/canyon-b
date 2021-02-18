"""
Implementation of the Bayesian Multi-Layer Percepton
"""


import torch.nn as nn
from blitz.modules import BayesianLinear
from topology import best_topo


class MLP_Bayesian(nn.Module):
    def __init__(self, top):
        input_size, n_hidden1, n_hidden2, output_size = best_topo[top]
        super(MLP_Bayesian, self).__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            BayesianLinear(input_size, n_hidden1),  # nn.Linear(input_size, n_hidden1), #
            nn.SELU(),  # nn.SELU(),  #  funzione bene con nn.ReLU(), ELU(),
            BayesianLinear(n_hidden1, n_hidden2),  # nn.Linear(n_hidden1, n_hidden2), #
            nn.SELU(),  # nn.SELU(), #
            BayesianLinear(n_hidden2, output_size),  # nn.Linear(n_hidden2, output_size), #
        )

    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.network(x)

