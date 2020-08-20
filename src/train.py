from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from data_loader import load_data
from utils import accuracy, one_hot_embedding # load_data, 
from model import GCNLPA

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default="citeseer",
                    help='Dataset name (cora, citeseer, pubmed).')
parser.add_argument('--layers', type=int, default=2,
                    help='Number of GCN-LPA layers.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--print-interval', type=int, default=20,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--Lambda', type=float, default=10,
                    help='Please refer to Eqn. (18) in original paper')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Load data

if args.dataset in ['cora', 'citeseer', 'pubmed']:
    data = load_data(args.dataset)
elif args.dataset in ['coauthor-cs', 'coauthor-phy']:
    data = load_npz(args.dataset)
else:
    n_nodes = 1000
    data = load_random(n_nodes=n_nodes, n_train=100, n_val=200, p=10/n_nodes)

features, labels, adj, idx_train, idx_val, idx_test = [data[i] for i in range(6)]



# adj, features, labels, idx_train, idx_val, idx_test = load_data()

features = features.to(device)
adj = adj.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)
labels_for_lpa = one_hot_embedding(labels, labels.max().item() + 1).type(torch.FloatTensor).to(device)

# Model and optimizer
model = GCNLPA(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            adj=adj,
            dropout_rate=args.dropout,
            layers=args.layers)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# if args.cuda:
model.to(device)
best_acc = 0.0

def train(epoch, best_acc):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, y_hat = model(features, labels_for_lpa)
    loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
    loss_lpa = F.nll_loss(y_hat, labels)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train = loss_gcn + args.Lambda * loss_lpa
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + \
    #     args.Lambda * F.nll_loss(y_hat, labels)
    loss_train.backward(retain_graph=True) # 
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output, _ = model(features, labels_for_lpa)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    if epoch % args.print_interval == (args.print_interval-1):
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'loss_test: {:.4f}'.format(loss_test.item()),
            'acc_test: {:.4f}'.format(acc_test.item()),
            'time: {:.4f}s'.format(time.time() - t))
    best_acc = acc_test.item() if acc_test.item() > best_acc else best_acc

    return best_acc




# def test():
#     model.eval()
#     output, _ = model(features, labels_for_lpa)
    
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    best_acc = train(epoch, best_acc)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print("Best acc: {:.4f}".format(best_acc))

# Testing
# test()