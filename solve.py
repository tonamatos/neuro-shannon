import torch
from torch import Tensor
from torch.optim import Adam
import numpy as np
import scipy
import time
from torch_sparse import SparseTensor
from models.model import MISGNNSolver, MISGNNEmbedding
from utils.dataset import MISDataset, load_from_directory
from utils.qubo import QUBO
from utils.dataloader import dataloader
from utils.metrics import SOLVED_ACCURACY, AVG_SIZE, TIME
import logging

class Solve():
    def __init__(self, args, data=None) -> None:
        self.args = args
        self.supervised = args.supervised
        self.qubo = args.qubo
        self.DGA = args.DGA
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.dropout = args.dropout_frac
        self.qubo_model = QUBO(args.p1, args.p2, args.n)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        if self.supervised:
            self.model = MISGNNEmbedding(1, args.hidden_size, args.num_hidden_layers, 0).to(self.device)
            self.model.load_state_dict(torch.load(args.pretrained, map_location=self.device))
            self.model.eval()
            if data is not None:
                self.input = data
            else:
                raise ValueError("Test data file path required!")
        else:
            self.input = args.input

        # Setup logging
        logging.basicConfig(filename='results.log', level=logging.INFO, 
                            format='%(asctime)s %(message)s', 
                            datefmt='%m/%d/%Y %I:%M:%S %p')
    
    def initialize_solver_model(self, embedding_d0, embedding_d1, num_classes):
        solver_model = MISGNNSolver(embedding_d0, embedding_d1, num_classes).to(self.device)
        optimizer = Adam(params=solver_model.parameters(), lr=self.learning_rate)
        return solver_model, optimizer

    def run(self):
        dataset = MISDataset(load_from_directory(self.input), k=self.args.k, supervised=self.supervised)
        dl = dataloader(dataset, self.batch_size)  # Using multiple workers to speed up data loading
        qubo_accuracy = SOLVED_ACCURACY()
        DGA_accuracy = SOLVED_ACCURACY()
        qubo_size = AVG_SIZE()
        DGA_size = AVG_SIZE()
        gt_size = AVG_SIZE()
        qubo_time = TIME()
        DGA_time = TIME()
        supervised_accuracy = SOLVED_ACCURACY()
        supervised_size = AVG_SIZE()
        supervised_time = TIME()

        for idx, graph in enumerate(dl):
            x: Tensor = graph.x.to(self.device)
            adj: SparseTensor = graph.adj.to(self.device)
            edge_index: Tensor = graph.edge_index.to(self.device)
            y = graph.y.to(self.device)
            
            node_emb = None

            if self.supervised:
                tik = time.time()
                embedding_d0 = 1
                embedding_d1 = self.args.d1
                probs = self.model(x.view(adj.size(0), -1), adj)
                adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones_like(edge_index[1], dtype=torch.float32), sparse_sizes=(adj.size(0), adj.size(0)))
                sol = mis_decode_np(adj, x, 1, 0, probs.squeeze(1))
                supervised_time.update(time.time()-tik)
                supervised_size.update(sol.sum(), self.args.batch_size)
                supervised_accuracy.update(sol.sum(), y.sum(), self.args.batch_size)
                node_emb = torch.tensor(sol).view(adj.size(0), -1).to(dtype=torch.float32, device=self.device)
            
            if self.qubo:
                embedding_d0 = 1
                embedding_d1 = self.args.d1
                if node_emb is None:
                    node_emb = x.view(adj.size(0), -1)
                num_classes = 1
                Q_matrix = self.qubo_model.create_Q_matrix(edge_index.cpu(), adj.size(0), x).to(self.device)
                rerun = True
                count = 0
                while rerun:
                    solver_model, optimizer = self.initialize_solver_model(embedding_d0, embedding_d1, num_classes)
                    output = solver_model(node_emb, adj)
                    loss = self.qubo_model.qubo_approx_cost(output, Q_matrix)
                    while loss.item() >= 0:  # Reinitialize if loss is non-ideal
                        solver_model, optimizer = self.initialize_solver_model(embedding_d0, embedding_d1, num_classes)
                        output = solver_model(node_emb, adj)
                        loss = self.qubo_model.qubo_approx_cost(output, Q_matrix)
                    tik = time.time()
                    for _ in range(self.num_epochs):
                        solver_model.train()
                        output = solver_model(node_emb, adj)
                        loss = self.qubo_model.qubo_approx_cost(output, Q_matrix)
                        
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()
                    if loss.item() < -self.args.penalty_threshold:
                        rerun = False
                    count += 1
                    if count >= 100:
                        print(f"Graph No. {idx} Hard to lower than threshold: {loss.item()}")
                        break
                if loss.item() < -self.args.penalty_threshold:
                    if idx % 10 == 0:
                        print(f"Graph No. {idx} Loss: {loss.item()}")
                    solver_model.eval()
                    sol1 = mis_decode_np(adj, x, self.args.c1, self.args.c2, solver_model(node_emb, adj).squeeze(1))
                    time_spent = time.time()-tik
                    qubo_time.update(time_spent)
                    qubo_size.update(sol1.sum(), self.args.batch_size)
                    qubo_accuracy.update(sol1.sum(), y.sum(), self.args.batch_size)

            if self.DGA:
                tik = time.time()
                sol2 = mis_decode_np(adj, x, 0, 1)
                DGA_time.update(time.time()-tik)
                DGA_size.update(sol2.sum(), self.args.batch_size)
                DGA_accuracy.update(sol2.sum(), y.sum(), self.args.batch_size)

            # Update Ground Truth data
            gt_size.update(y.sum(), self.args.batch_size)
        
        if self.qubo or self.supervised:
            if self.supervised:
                logging.info(f"Supervised + GD size: {supervised_size.avg}, {supervised_size.size}, Time: {supervised_time.avg_time()}")
            if self.qubo:
                logging.info(f"QUBO size: {qubo_size.avg}, {qubo_size.size}, QUBO Solved Accuracy: {qubo_accuracy.accuracy()}, Time: {qubo_time.avg_time()}")
            
        if self.DGA:
            logging.info(f"DGA size: {DGA_size.avg}, {DGA_size.size}, DGA Solved Accuracy: {DGA_accuracy.accuracy()}, Time: {DGA_time.avg_time()}")
        
        logging.info(f"Ground Truth size: {gt_size.avg}, {gt_size.size}")

def get_classification(output):
    classification = (output >= 0.5).int()
    return classification.cpu().numpy().ravel()

def mis_decode_np(adj_matrix, normalized_node_embeddings, c1=0, c2=1, predictions=None):
    if c1 >= 1:
        predictions = predictions.data
    solution = np.zeros(adj_matrix.size(0))
    if c1 == 0:
        combined_scores = np.asarray(normalized_node_embeddings)
    else:
        combined_scores = c1 * predictions.cpu() + c2 * np.asarray(normalized_node_embeddings)
    sorted_predict_labels = np.argsort(-combined_scores)
    csr_adj_matrix = scipy.sparse.csr_matrix(adj_matrix.to_dense().cpu().numpy())

    for i in sorted_predict_labels:
        if solution[i] == -1:
            continue
        solution[csr_adj_matrix[i].nonzero()[1]] = -1
        solution[i] = 1

    return (solution == 1).astype(int)
