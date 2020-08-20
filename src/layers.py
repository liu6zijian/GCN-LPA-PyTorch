import math

import torch

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F


class GCNLPAConv(nn.Module):
    """
    A GCN-LPA layer. Please refer to: https://arxiv.org/abs/2002.06755
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(GCNLPAConv, self).__init__()
        self.adj = adj
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        # adj_mask = torch.where(adj>0, adj_mask, -9e9*torch.ones_like(adj))
        # adj_mask = torch.softmax(adj_mask, dim=1)
        # self.adjacency_mask = adj_mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, y, adj_mask):
        # adj = adj.to_dense()
        # W * x
        mask = torch.where(self.adj>0, adj_mask, -9e9*torch.ones_like(adj_mask))
        adj = torch.softmax(mask, dim=1)
        support = torch.mm(x, self.weight)
        # # Hadamard Product: A' = Hadamard(A, M)
        # adj = self.adj * mask
        # # Row-Normalize: D^-1 * (A')
        # adj = F.normalize(adj, p=1, dim=1)
        # # adj = F.softmax(adj, dim=1)

        # output = D^-1 * A' * X * W
        output = torch.mm(adj, support)
        # y' = D^-1 * A' * y
        y_hat = torch.mm(adj, y)
        if self.bias is not None:
            return output + self.bias, y_hat
        else:
            return output, y_hat

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
