"""MIS (Maximal Independent Set) dataset."""

import os
import pickle
from utils.qubo import QUBO

import numpy as np
import torch

from torch_geometric.data import Data as GraphData
from torch_sparse import SparseTensor

from sklearn.model_selection import train_test_split

class MISDataset(torch.utils.data.Dataset):
  def __init__(self, file_lines):
    self.file_lines = file_lines
    # self.qubo = QUBO(p1, p2)
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
    node_embeddings = np.ones((num_nodes, 1), dtype=np.float32)

    # q_indices, values = self.qubo.create_Q_matrix(edges=edge_index, num_nodes=num_nodes)

    edge_index_tensor = torch.from_numpy(edge_index).long()
    # q_index_tensor = torch.from_numpy(q_indices).long() 
    # Q_matrix = SparseTensor(
    #     row=q_index_tensor[0],
    #     col=q_index_tensor[1],
    #     value=values,
    #     sparse_sizes=(num_nodes, num_nodes),
    # )
    adj_matrix = SparseTensor(
        row=edge_index_tensor[0],
        col=edge_index_tensor[1],
        value=torch.ones_like(edge_index_tensor[0].float()),
        sparse_sizes=(num_nodes, num_nodes),
    )
    graph_data = GraphData(x=torch.from_numpy(node_embeddings),
                           edge_index=edge_index_tensor,
                           adj=adj_matrix,
                           y=torch.from_numpy(node_labels))

    return graph_data
  
def train_val_split(data_directory, test_size=0.2, random_state=1999):
    all_files = load_from_directory(data_directory)

    # Split the files into training and validation sets
    train_files, val_files = train_test_split(all_files, test_size=test_size, random_state=random_state)

    return train_files, val_files

def load_from_directory(data_directory):
   all_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.gpickle')]
   return all_files
