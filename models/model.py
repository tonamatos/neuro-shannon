from .layers import GraphConvolutionLayer

import torch
import torch.nn as nn
import torch.nn.functional as F

class CombGNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_classes, dropout_frac=0.0):
        super(CombGNN, self).__init__()
        self.conv1 = GraphConvolutionLayer(embedding_size, hidden_size)
        self.conv2 = GraphConvolutionLayer(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_frac)

    def forward(self, node_feats, adj_matrix, training=False):
        h = F.relu(self.conv1(node_feats, adj_matrix))
        if training:
            h = self.dropout(h)
        h = self.conv2(h, adj_matrix)
        probs = torch.sigmoid(h)
        return probs
