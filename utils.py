# This script contains the necessary utils for executing the CBO
import numpy as np
import random
import torch

# get noise matrix for SDE without variance scale
def inplace_randn(size, device=None):
    # INPUT: - size (size-tensor(2x2)): size of the input agents (n_batches/N x d) 
    #        - device (device): represents where torch.tensors will be allocated
    # OUTPUT: - noise ((n_batches/N x d))-tensor-matrix): standard normal noise matrix
    device = torch.device('cpu') if device is None else device
    # https://discuss.pytorch.org/t/random-number-generation-speed/12209
    if device.type == 'cuda':
        return torch.cuda.FloatTensor(*size).normal_(0, 1)
    return torch.FloatTensor(*size).normal_(0, 1)
