import torch
from torch import Tensor
from torch.optim import Adam
import torch.nn as nn
from torch_geometric.loader import DataLoader

from models.model import CombGNN
from utils.loss_track import AverageMeter
from utils.dataset import MISDataset, train_val_split, load_from_directory
from utils.qubo import QUBO
from utils.dataloader import dataloader

class Solve():
    def __init__(self, args, data=None) -> None:
        self.supervised = args.supervised
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.dropout = args.dropout_frac
        self.qubo = QUBO(args.p1, args.p2)
        
        if self.supervised:
            if data is not None:
                self.input = data
            else:
                raise ValueError("Test data file path required!")
        else:
            self.input = args.input
        
    def run(self):
        dataset = MISDataset(load_from_directory(self.input))
        dl = dataloader(dataset, self.batch_size)

        for graph in dl:
            adj = graph.adj
            edge_index: Tensor = graph.edge_index

            if self.supervised:
                embedding_d0 = 1
                embedding_d1 = 10
                node_emb = graph.x
            else:
                embedding_d0 = adj.size(0)
                embedding_d1 = embedding_d0 // 2
                node_emb = adj

            num_classes = 1
            model = CombGNN(embedding_d0, embedding_d1, num_classes, self.dropout)
            optimizer = Adam(params=model.parameters(), lr=self.learning_rate)
            Q_matrix = self.qubo.create_Q_matrix(edge_index, adj.size(0))
            
            for epoch in range(self.num_epochs):
                model.train()
                output = model(node_emb, adj)
                # print(output)
                
                # Create Q_matrix only once for each graph
                Q_matrix = self.qubo.create_Q_matrix(edge_index, adj.size(0))
                
                # Solving
                loss = self.qubo.qubo_approx_cost(output, Q_matrix)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch+1}: {loss.item()}")
            
            print("Solution is:",output)
            import time
            time.sleep(10)
            

