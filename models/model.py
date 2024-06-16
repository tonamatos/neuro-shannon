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
        self.last_layer = GraphConvolutionLayer(hidden_size, hidden_size//2)
        self.fc = nn.Linear(hidden_size//2, 5)
        self.solver_layer = GraphConvolutionLayer(5, 1)
        self.dropout = nn.Dropout(dropout_frac)
        self.training = training
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(num_hidden_layers)])

    def forward(self, node_features, adj_matrix):
        h = F.relu(self.first_layer(node_features, adj_matrix))
        for i, layer in enumerate(self.hidden_layers):
            h = layer(h, adj_matrix)
            h = self.batch_norms[i](h)
            if self.training:
                h = self.dropout(h)
        h = F.tanh(self.last_layer(h, adj_matrix))
        h = F.tanh(self.fc(h))
        probs = torch.sigmoid(self.solver_layer(h, adj_matrix))
        return probs