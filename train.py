# Package Import here
import torch
from torch.optim import Adam
import torch.nn as nn
import time

class Train():
    def __init__(self, args) -> None:
        self.model = ###
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_dataset = MISDataset()
        self.validation_dataset = MISDataset()
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.dropout_frac = args.dropout_frac

    def DataLoader(self):
        # TODO: Load dataset into batch
        pass

    def train(self):        
        self.model.train()
        # load train data as dataloader
        train_dataloader = self.DataLoader(self.train_dataset)
        optimizer = Adam(params=self.model.parameters(), lr=self.learning_rate)
        l1 = nn.CrossEntropyLoss()
        
        # Start training
        for epoch in range(self.num_epochs + 1):
            tik = time.time()
            for graph, label in train_dataloader:
                graph, target = graph.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                initial_emb, adj = graph
                output = self.model(initial_emb, adj)
                loss = l1(output, target) # + qubo_approx_cost(output, Q)
                loss.backward()
                optimizer.step()
                loss_track.update()
            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.5f" % (epoch + 1, self.num_epochs, elapse, loss_track.avg))





