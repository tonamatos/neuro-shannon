# Package Import here
import torch
from torch.optim import Adam
import torch.nn as nn

from models.model import MISGNNEmbedding
from utils.loss_track import AverageMeter
from utils.dataset import MISDataset, train_val_split
from utils.qubo import QUBO
from utils.dataloader import dataloader

import time

class Train:
    def __init__(self, args) -> None:
        self.model = MISGNNEmbedding(1, args.hidden_size, args.num_hidden_layers, args.dropout_frac, training=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_data, self.val_data = train_val_split(args.input)
        self.train_dataset = MISDataset(self.train_data, supervised=True)
        self.validation_dataset = MISDataset(self.val_data, supervised=True)
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.num_hidden_layers = args.num_hidden_layers
        self.input = args.input
        self.output = args.output
        self.loss_track = AverageMeter()
        self.qubo = QUBO(args.p1, args.p2)
    
    def train(self):        
        self.model.train()
        # load train data as dataloader
        print(self.device)
        train_dataloader = dataloader(self.train_dataset, self.batch_size)
        val_dataloader = dataloader(self.validation_dataset, self.batch_size)
        optimizer = Adam(params=self.model.parameters(), lr=self.learning_rate)
        l1 = nn.BCELoss()
        
        # Start training
        for epoch in range(self.num_epochs):
            tik = time.time()
            self.loss_track.reset()
            for graph in train_dataloader:
                initial_emb = graph.x
                label = graph.y
                edge_index = graph.edge_index
                adj_matrix = graph.adj
                initial_emb, label, edge_index= initial_emb.to(self.device), label.to(self.device), edge_index.to(self.device)              
                optimizer.zero_grad()
                probs = self.model(initial_emb.view(adj_matrix.size(0), -1), adj_matrix)
                loss = l1(probs.squeeze(1), label.float()) 
                loss.backward()
                optimizer.step()
                self.loss_track.update(val=loss.item())
            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.5f" % (epoch + 1, self.num_epochs, elapse, self.loss_track.avg))
        
        # Validation
        self.loss_track.reset()
        self.model.eval()
        for val_g in val_dataloader:
            initial_emb = val_g.x
            label = val_g.y
            edge_index = val_g.edge_index
            adj_matrix = val_g.adj
            emb, val_probs = self.model(initial_emb, adj_matrix)
            loss = l1(val_probs.squeeze(1), label.float())
            self.loss_track.update(val=loss.item())
            print(emb)
            print(val_probs)
            print(label)
        print("Validation Complete; Loss: %.5f" % (self.loss_track.avg))
        time.sleep(10)

        # Save to the directory if I satisfy with my result
        print("Training completed, saving model to %s." % self.output)
        torch.save(self.model.state_dict(), self.output / f"final_model_{str(self.input).split('/')[-1]}_epoch{self.num_epochs}_num_hidden{self.num_hidden_layers}.torch")
        

