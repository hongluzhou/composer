import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


def build_mlp(input_dim, hidden_dims, output_dim=None, use_batchnorm=False, dropout=0):
    layers = []
    D = input_dim
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    if hidden_dims:
        for dim in hidden_dims:
            layers.append(nn.Linear(D, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU(inplace=True))
            D = dim
    if output_dim:
        layers.append(nn.Linear(D, output_dim))
    return nn.Sequential(*layers)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        support = torch.bmm(adj, hidden)
        output = self.act(support)
        return output, adj

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )
    
    
def get_joint_graph(num_nodes=17, joint_graph_path='joint_graph.txt'):
    adj = np.zeros((num_nodes, num_nodes))
    with open(joint_graph_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        adj[int(line.split()[0]), int(line.split()[1])] = 1
        adj[int(line.split()[1]), int(line.split()[0])] = 1
    # adj = sp.csr_matrix(adj)
    return torch.Tensor(adj)