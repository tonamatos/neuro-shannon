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
from utils.metrics import SOLVED_ACCURACY, AVG_SIZE, AVG_DROP, AVG_TIME

class Solve():
    def __init__(self, args, data=None) -> None:
        self.args = args
        self.supervised = args.supervised
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.dropout = args.dropout_frac
        self.qubo = QUBO(args.p1, args.p2)
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
    
    def initialize_solver_model(self, embedding_d0, embedding_d1, num_classes):
        solver_model = MISGNNSolver(embedding_d0, embedding_d1, num_classes).to(self.device)
        optimizer = Adam(params=solver_model.parameters(), lr=self.learning_rate)
        return solver_model, optimizer
    
    def run(self):
        dataset = MISDataset(load_from_directory(self.input), supervised=self.supervised)
        dl = dataloader(dataset, self.batch_size)
        qubo_accuracy = SOLVED_ACCURACY()
        DGA_accuracy = SOLVED_ACCURACY()
        qubo_size = AVG_SIZE()
        DGA_size = AVG_SIZE()
        gt_size = AVG_SIZE()
        qubo_drop = AVG_DROP()
        DGA_drop = AVG_DROP()
        qubo_time = AVG_TIME()
        DGA_time = AVG_TIME()
        

        for idx, graph in enumerate(dl):
            x: Tensor = graph.x.to(self.device)
            adj: SparseTensor = graph.adj.to(self.device)
            edge_index: Tensor = graph.edge_index.to(self.device)
            y = graph.y.to(self.device)
            if self.supervised:
                embedding_d0 = 1
                embedding_d1 = self.args.d1
                normalized_node_degree = x
                probs = self.model(x.view(adj.size(0), -1), adj)
                print(probs)
                print(torch.nn.BCELoss()(probs.squeeze(1), y.float()))
                adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones_like(edge_index[1], dtype=torch.float32), sparse_sizes=(adj.size(0), adj.size(0)))
                sol = mis_decode_np(probs.squeeze(1), adj, normalized_node_degree, 1, 0)
                print("GNN Solution is:", sol, "Capacity is ", sol.sum())
                node_emb = torch.tensor(sol).view(adj.size(0), -1).to(dtype=torch.float32, device=self.device)
            else:
                embedding_d0 = 1
                embedding_d1 = self.args.d1
                # x has already the degree-based initialized
                node_emb = x.view(adj.size(0), -1)
                normalized_node_degree = x

            num_classes = 1
            Q_matrix = self.qubo.create_Q_matrix(edge_index.cpu(), adj.size(0), normalized_node_degree).to(self.device)
            rerun = True
            while rerun:
                tik = time.time()
                solver_model, optimizer = self.initialize_solver_model(embedding_d0, embedding_d1, num_classes)
                for _ in range(self.num_epochs):
                    solver_model.train()
                    output = solver_model(node_emb, adj)
                    
                    # Solving
                    loss = self.qubo.qubo_approx_cost(output, Q_matrix)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    # print(f"Epoch {epoch+1}: {loss.item()}")
                if loss.item() < -self.args.penalty_threshold:
                    rerun = False
            if idx % 10 == 0:
                print(f"Graph No. {idx} Loss:{loss.item()}")
            solver_model.eval()
            # QUBO Solution evaluation
            sol1 = mis_decode_np(adj, normalized_node_degree, self.args.c1, self.args.c2, solver_model(node_emb, adj).squeeze(1))
            qubo_time.update(time.time()-tik, self.args.batch_size)
            qubo_size.update(sol1.sum(), self.args.batch_size)
            qubo_accuracy.update(sol1.sum(), y.sum(), self.args.batch_size)
            qubo_drop.update(sol1.sum(), y.sum(), self.args.batch_size)

            # DGA Solution Evaluation
            tik = time.time()
            sol2 = mis_decode_np(adj, normalized_node_degree, 0, 1)
            DGA_time.update(time.time()-tik, self.args.batch_size)
            DGA_size.update(sol2.sum(), self.args.batch_size)
            DGA_accuracy.update(sol2.sum(), y.sum(), self.args.batch_size)
            DGA_drop.update(sol2.sum(), y.sum(), self.args.batch_size)
            gt_size.update(y.sum(), self.args.batch_size)
        print("QUBO size: ", qubo_size.avg, qubo_size.size, "QUBO Solved Accuracy: ", qubo_accuracy.accuracy(), "QUBO Drop: ", qubo_drop.drop_percentage())
        print("DGA size: ", DGA_size.avg, DGA_size.size, "DGA Solved Accuracy: ", DGA_accuracy.accuracy(), "DGA Drop:", DGA_drop.drop_percentage())
        print("Gound Truth size: ", gt_size.avg, gt_size.size)


def get_classification(output):
    classification = (output >= 0.5).int()
    return classification.cpu().numpy().ravel()

def mis_decode_np(adj_matrix, normalized_node_embeddings, c1=0, c2=1, predictions=None, ):
    """Decode the labels to the MIS."""
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

def verify_sol(solution, adj) -> bool:
    independent_set_index = []
    for i in range(len(solution)):
        if solution[i] == 1:
            independent_set_index.append(i)
    for i in range(len(independent_set_index) - 1):
        for j in range(i + 1, len(independent_set_index)):
            if adj[independent_set_index[i]][independent_set_index[j]] != 0:
                print("These are connected edges:", independent_set_index[i], independent_set_index[j])
                return False
    for i in range(len(solution)):
        if solution[i] == 0:
            can_add = True
            for idx in independent_set_index:
                if adj[i][idx] != 0:
                    can_add = False
                    break
            if can_add:
                return False 
    return True
