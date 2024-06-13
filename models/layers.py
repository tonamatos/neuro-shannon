import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolutionLayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: Tensor | SparseTensor, adj_matrix: Tensor | SparseTensor):
        # print("Input", input.to_dense())
        support = torch.mm(input, self.weight)
        
        output = torch.spmm(adj_matrix.to_dense(), support)
        # print("output (Dense):", output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self): 
        return self.__class__.__name__ + ' ('\
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class ResidualGraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualGraphConvolutionLayer, self).__init__()
        self.conv1 = GraphConvolutionLayer(in_features, out_features)
        self.conv2 = GraphConvolutionLayer(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x, adj_matrix):
        identity = x

        out = self.conv1(x, adj_matrix)
        out = self.bn(out)
        out = F.relu(out)

        out = self.conv2(out, adj_matrix)
        out = self.bn(out)

        out += identity
        out = F.relu(out)

        return out