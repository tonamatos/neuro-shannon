from .layers import GraphConvolutionLayer, ResidualGraphConvolutionLayer

import torch
import torch.nn as nn
import torch.nn.functional as F

class MISGNNSolver(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_classes):
        super(MISGNNSolver, self).__init__()
        self.conv1 = GraphConvolutionLayer(embedding_size, hidden_size)
        self.conv2 = GraphConvolutionLayer(hidden_size, num_classes)

    def forward(self, node_feats, adj_matrix):
        h = F.relu(self.conv1(node_feats, adj_matrix))
        h = self.conv2(h, adj_matrix)
        probs = torch.sigmoid(h)
        return probs

class MISGNNEmbedding(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_hidden_layers, dropout_frac, training=False):
        super(MISGNNEmbedding, self).__init__()
        self.first_layer = GraphConvolutionLayer(embedding_size, hidden_size)
        self.hidden_layers = nn.ModuleList([ResidualGraphConvolutionLayer(hidden_size, hidden_size) for _ in range(num_hidden_layers)])
        self.solver_layer = GraphConvolutionLayer(hidden_size, 1)
        self.num_hidden_layers = num_hidden_layers
        self.dropout = nn.Dropout(dropout_frac)
        self.training = training

    def forward(self, node_features, adj_matrix):
        h = F.relu(self.first_layer(node_features, adj_matrix))
        for layer in self.hidden_layers:
            h = layer(h, adj_matrix)
            if self.training:
                h = self.dropout(h)
        probs = torch.sigmoid(self.solver_layer(h, adj_matrix))
        return probs