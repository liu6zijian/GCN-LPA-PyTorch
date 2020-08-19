import torch.nn as nn
import torch.nn.functional as F
from layers import GCNLPAConv

class GCNLPA(nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj, dropout_rate):
        super(GCNLPA, self).__init__()

        self.gc1 = GCNLPAConv(nfeat, nhid, adj)
        self.gc2 = GCNLPAConv(nhid, nclass, adj)
        self.dropout = dropout_rate

    def forward(self, x, y):
        x, y_hat = self.gc1(x, y)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x, y_hat = self.gc2(x, y_hat)
        return F.log_softmax(x, dim=1), F.log_softmax(y_hat,dim=1)