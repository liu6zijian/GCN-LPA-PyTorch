import torch.nn as nn
import torch.nn.functional as F
from layers import GCNLPAConv
import torch
class GCNLPA(nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj, dropout_rate, layers):
        super(GCNLPA, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLPAConv(nfeat, nhid, adj))
        for _ in range(layers-2):
            self.layers.append(GCNLPAConv(nhid, nhid, adj))
        
        self.layers.append(GCNLPAConv(nhid, nclass, adj))
        self.dropout = dropout_rate
        self.adj_mask = nn.Parameter(adj.clone())

    def forward(self, x, y):
        yhat = y
        for net in self.layers[:-1]:
            x = F.dropout(x, self.dropout, training=self.training)
            x, y_hat = net(x, y, self.adj_mask)
            x = F.relu(x)

        x = F.dropout(x, self.dropout, training=self.training)
        x, y_hat = self.layers[-1](x, y_hat, self.adj_mask)
        return F.log_softmax(x, dim=1), F.log_softmax(y_hat,dim=1)