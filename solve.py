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

class Solve():
    def __init__(self, args, data=None) -> None:
        self.args = args
        self.supervised = args.supervised
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.dropout = args.dropout_frac
        self.qubo = QUBO(args.p1, args.p2)
        self.model = None
        
        if self.supervised:
            self.model = MISGNNEmbedding(1, args.hidden_size, args.num_hidden_layers, 0)
            self.model.load_state_dict(torch.load(args.pretrained))
            self.model.eval()
            if data is not None:
                self.input = data
            else:
                raise ValueError("Test data file path required!")
        else:
            self.input = args.input
    
    def initialize_solver_model(self, embedding_d0, embedding_d1, num_classes):
        solver_model = MISGNNSolver(embedding_d0, embedding_d1, num_classes)
        optimizer = Adam(params=solver_model.parameters(), lr=self.learning_rate)
        return solver_model, optimizer
    
    def run(self):
        dataset = MISDataset(load_from_directory(self.input), supervised=self.supervised)
        dl = dataloader(dataset, self.batch_size)

        for graph in dl:
            x: Tensor = graph.x
            adj: SparseTensor = graph.adj
            edge_index: Tensor = graph.edge_index
            y = graph.y
              
            if self.supervised:
                embedding_d0 = 1
                embedding_d1 = 100
                probs = self.model(x.view(adj.size(0), -1), adj)
                adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones_like(edge_index[1], dtype=torch.float32), sparse_sizes=(len(_), len(_)))
                sol = mis_decode_np(probs.squeeze(1), adj, x, 1, 6)
                print("GNN Solution is:", sol, "Capacity is ", sol.sum())
                node_emb = torch.tensor(sol).view(adj.size(0), -1).to(dtype=torch.float32)
            else:
                embedding_d0 = 1
                embedding_d1 = 100
                node_emb = x.view(adj.size(0), -1)

            num_classes = 1
            Q_matrix = self.qubo.create_Q_matrix(edge_index, adj.size(0), x.data)
            rerun = True
            while rerun:
                solver_model, optimizer = self.initialize_solver_model(embedding_d0, embedding_d1, num_classes)
                for epoch in range(self.num_epochs):
                    solver_model.train()
                    output = solver_model(node_emb, adj)
                    
                    # Solving
                    loss = self.qubo.qubo_approx_cost(output, Q_matrix)
                    if loss.item() >= 0:
                        solver_model, optimizer = self.initialize_solver_model(embedding_d0, embedding_d1, num_classes)
                        continue

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    # print(f"Epoch {epoch+1}: {loss.item()}")
                if loss.item() < -1000:
                    rerun = False
            print(f"Loss:{loss.item()}")
            solver_model.eval()
            sol1 = mis_decode_np(solver_model(node_emb, adj).squeeze(1), adj, x.data, 1, 6)
            print("Solution is:", sol1, "Capacity is ", sol1.sum())
            sol2 = mis_decode_np(get_classification(solver_model(node_emb, adj).squeeze(1)), adj, x.data, 0, 1)
            print("Greedy Solution is:", sol2, "Capacity is ", sol2.sum())
            print("Gound Truth", y, "Capacity is ", y.sum())
            # print("The solution is: ", verify_sol(sol, adj.to_dense()), "Ground Truth solutioin is: ", verify_sol(y, adj.to_dense()))
            time.sleep(1)
            

def get_classification(output):
    classification = (output >= 0.5).int()
    return classification.numpy().ravel()

def mis_decode_np(predictions, adj_matrix, normalized_node_embeddings, c1=0, c2=1):
    """Decode the labels to the MIS."""
    if c1 >= 1:
        predictions = predictions.data
    solution = np.zeros_like(predictions)
    # print(predictions, np.asarray(normalized_node_embeddings))
    combined_scores = c1 * predictions + c2 * np.asarray(normalized_node_embeddings)
    # print(combined_scores)
    sorted_predict_labels = np.argsort(-combined_scores)
    csr_adj_matrix = scipy.sparse.csr_matrix(adj_matrix.to_dense().numpy())

    for i in sorted_predict_labels:

        if solution[i] == -1:
            continue

        solution[csr_adj_matrix[i].nonzero()[1]] = -1
        solution[i] = 1

    return (solution == 1).astype(int)

def verify_sol(solution, adj) -> bool:
    independent_set_index = []
    # Get index of 1 in solution
    for i in range(len(solution)):
        if solution[i] == 1:
            independent_set_index.append(i)
    # Check if the set is independent
    for i in range(len(independent_set_index)-1):
        for j in range(i + 1, len(independent_set_index)):
            if adj[independent_set_index[i]][independent_set_index[j]] != 0:
                print("These are connected edges:", independent_set_index[i], independent_set_index[j])
                return False
    # Check if the solution is maximal
    for i in range(len(solution)):
        if solution[i] == 0:
            # Check if the vertex can be added without breaking the independence
            can_add = True
            for idx in independent_set_index:
                if adj[i][idx] != 0:
                    can_add = False
                    break
            if can_add:
                return False 
    return True