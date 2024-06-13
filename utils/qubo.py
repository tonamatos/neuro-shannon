from pyqubo import Array
import numpy as np
import torch
from torch_sparse import SparseTensor
from torch_sparse import spmm

class QUBO:
    def __init__(self, p1, p2) -> None:
        super(QUBO, self).__init__()
        self.p1 = p1
        self.p2 = p2

    def create_mis_model(self, edges, num_nodes):
        X = Array.create("X", shape=(num_nodes,), vartype="BINARY")

        hamiltonian = -self.p1 * sum(X)
        for i in range(len(edges[0])):
            u = edges[0][i].item()
            v = edges[1][i].item()

            hamiltonian += self.p2*(X[u] * X[v])
        return hamiltonian.compile()
    
    def create_Q_matrix(self, edges, num_nodes, normalized_node_degrees):
        indices = []
        values = []
        p1 = self.p2 * len(edges[0]) / (num_nodes)

        # Adding diagonal elements with normalized penalties
        for i in range(num_nodes):
            indices.append([i, i])
            # Scale penalty to a reasonable range, e.g., [-self.p2, 0]
            penalty = -p1 * (normalized_node_degrees[i]**(4))
            values.append(penalty)

        for i in range(len(edges[0])):
            u = edges[0][i].item()
            v = edges[1][i].item()
            if u != v:
                indices.append([u, v])
                values.append(self.p2)
        # Convert lists to tensors
        indices = torch.tensor(indices, dtype=torch.int64).t()
        values = torch.tensor(values, dtype=torch.float32)

        Q_matrix = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
        return Q_matrix
    
    def qubo_approx_cost(self, output, Q: torch.Tensor):
        # Perform sparse matrix multiplication Q * output
        Q_output = torch.spmm(Q, output)

        # Compute the final cost: output.T * (Q * output)
        cost = torch.matmul(output.T, Q_output)
        return cost
