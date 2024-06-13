"""MIS (Maximal Independent Set) dataset."""

import os
import pickle
from utils.qubo import QUBO

import numpy as np
import torch

from torch_geometric.data import Data as GraphData
from torch_sparse import SparseTensor
import scipy.sparse as sp

from sklearn.model_selection import train_test_split

class MISDataset(torch.utils.data.Dataset):
  def __init__(self, file_lines, supervised=False):
    self.file_lines = file_lines
    self.supervised = supervised
    print(f'Loaded with {len(self.file_lines)} examples')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    with open(self.file_lines[idx], "rb") as f:
      graph = pickle.load(f)

    num_nodes = graph.number_of_nodes()
    node_labels = [_[1] for _ in graph.nodes(data='label')]
    if node_labels is not None and node_labels[0] is not None:
        node_labels = np.array(node_labels, dtype=np.int64)
    else:
        node_labels = np.zeros(num_nodes, dtype=np.int64)
        node_labels[:] = -1
    assert node_labels.shape[0] == num_nodes
    
    edges = np.array(graph.edges, dtype=np.int64)
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    # # add self loop
    # self_loop = np.arange(num_nodes).reshape(-1, 1).repeat(2, axis=1)
    # edges = np.concatenate([edges, self_loop], axis=0)
    edges = edges.T

    return num_nodes, node_labels, edges

  def __getitem__(self, idx):
    num_nodes, node_labels, edge_index = self.get_example(idx)
    # Initial embeddings to all ones
    # node_embeddings = np.ones((num_nodes, 1), dtype=np.float32)

    normalized_degrees = normalize_node_degree_list(get_node_degree(edges=edge_index, num_nodes=num_nodes))
    # print(normalized_degrees)
    # import time
    # time.sleep(10)
    if self.supervised:
      adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                              shape=(num_nodes, num_nodes),
                              dtype=np.float32)

      # Build symmetric adjacency matrix
      adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

      # Add self-loops
      adj = adj + sp.eye(adj.shape[0])

      # Normalize the adjacency matrix
      adj = normalize_adj(adj)
      adj_coo = adj.tocoo()

      edge_index_tensor = torch.tensor(np.vstack((adj_coo.row, adj_coo.col)), dtype=torch.long)
      edge_weight_tensor = torch.tensor(adj_coo.data, dtype=torch.float32)
    else:
      edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
      edge_weight_tensor = torch.ones(edge_index_tensor.size(1), dtype=torch.float32)
    
    adj_matrix = SparseTensor(
        row=edge_index_tensor[0],
        col=edge_index_tensor[1],
        value=edge_weight_tensor,
        sparse_sizes=(num_nodes, num_nodes),
    )

    graph_data = GraphData(x=torch.from_numpy(normalized_degrees).to(torch.float32),
                           edge_index=edge_index_tensor,
                           adj=adj_matrix,
                           y=torch.from_numpy(node_labels))

    return graph_data
  
def train_val_split(data_directory, test_size=0.05, random_state=1999):
    all_files = load_from_directory(data_directory)

    # Split the files into training and validation sets
    train_files, val_files = train_test_split(all_files, test_size=test_size, random_state=random_state)

    return train_files, val_files

def load_from_directory(data_directory):
   all_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.gpickle')]
   return all_files

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def get_node_degree(edges, num_nodes):
    node_degree_list = [0 for _ in range(num_nodes)]

    for i in range(len(edges[0])):
        if edges[0][i] != edges[1][i]:
            node_degree_list[edges[0][i]] += 1

    return node_degree_list

def normalize_node_degree_list(node_degree_list, penalty=1):
    node_degree_array = np.array(node_degree_list)
    
    min_degree = np.min(node_degree_array)
    max_degree = np.max(node_degree_array)
    
    if max_degree - min_degree != 0:
        normalized_degrees = (node_degree_array - min_degree) / (max_degree - min_degree)
    else:
        normalized_degrees = np.full_like(node_degree_array, penalty, dtype=np.float32)
    
    normalized_degrees = 1 / (normalized_degrees + 1)
    
    return normalized_degrees
