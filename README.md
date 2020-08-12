# GCN-LPA-PyTorch
A PyTorch Implementation of GCN-LPA ([arXiv](https://arxiv.org/abs/2002.06755)):

> Unifying Graph Convolutional Neural Networks and Label Propagation  
> Hongwei Wang, Jure Leskovec  
> arXiv Preprint, 2020

GCN-LPA is an end-to-end model that unifies Graph Convolutional Neural Networks (GCN) and Label Propagation Algorithm (LPA) for adaptive semi-supervised node classification.

## TODO
This repo is going to be done...

## Prerequisites

The code has been tested running under Python 3.8.1, with the following packages installed (along with their dependencies):
- torch==
- scipy==
- numpy==

## Notification
- This is not an official implementation.
- Please cite the following papers if you use the code in your work:
```
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}

@article{wang2020unifying,
    title={Unifying Graph Convolutional Neural Networks and Label Propagation},
    author={Hongwei Wang and Jure Leskovec},
    journal={arXiv preprint arXiv:2002.06755}
    year={2020},
}
```