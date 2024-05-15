# Package Import here
import torch
from torch.optim import Adam
import torch.nn as nn
from torch_geometric.loader import DataLoader

from models.model import CombGNN
from utils.loss_track import AverageMeter
from utils.dataset import MISDataset, train_val_split

import time

class Train:
    def __init__(self, args) -> None:
        self.model = CombGNN(1, args.hidden_size, 1, args.dropout_frac)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_data, self.val_data = train_val_split(args.input)
        self.train_dataset = MISDataset(self.train_data)
        self.validation_dataset = MISDataset(self.val_data)
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.output = args.output
        self.loss_track = AverageMeter()

    def dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

    def train(self):        
        self.model.train()
        # load train data as dataloader
        print(self.device)
        train_dataloader = self.dataloader(self.train_dataset)
        optimizer = Adam(params=self.model.parameters(), lr=self.learning_rate)
        l1 = nn.MSELoss()
        
        # Start training
        for epoch in range(self.num_epochs + 1):
            tik = time.time()
            self.loss_track.reset()
            for graph in train_dataloader:
                initial_emb = graph.x
                label = graph.y
                edge_index = graph.edge_index
                adj = graph.adj

                initial_emb, label, edge_index, adj = initial_emb.to(self.device), label.to(self.device), edge_index.to(self.device), adj.to(self.device)
                optimizer.zero_grad()
                output = self.model(initial_emb, edge_index)
                # print("output", output.squeeze(1))
                # print(label)
                loss = l1(output.squeeze(1), label.float()) # + qubo_approx_cost(output, Q)
                print(loss)
                loss.backward()
                optimizer.step()
                self.loss_track.update(val=loss.item())
            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.5f" % (epoch + 1, self.num_epochs, elapse, self.loss_track.avg))
        
        print("Training completed, saving model to %s" % self.output)





