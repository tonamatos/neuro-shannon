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
    
    def create_Q_matrix(self, edges, num_nodes):
        indices = []
        values = []

        for i in range(num_nodes):
            indices.append([i,i])
            values.append(-self.p1)

        for i in range(len(edges[0])):
            u = edges[0][i].item()
            v = edges[1][i].item()
            if u != v:
                indices.append([u, v])
                values.append(self.p2)
        # Convert lists to tensors
        indices = np.array(indices, dtype=np.int64).T
        indices_tensor = torch.from_numpy(indices).long()
        # print(indices)
        values = torch.tensor(values, dtype=torch.float32)

        Q_matrix = SparseTensor(
            row=indices_tensor[0],
            col=indices_tensor[1],
            value=values,
            sparse_sizes=(num_nodes, num_nodes),
        )
        return Q_matrix
    
    def qubo_approx_cost(self, output, Q: SparseTensor):
        num_nodes = Q.sizes()[0]

        index = torch.stack([Q.storage.row(), Q.storage.col()])
        value = Q.storage.value()

        # Perform sparse matrix multiplication Q * output
        Q_output = spmm(index, value, num_nodes, num_nodes, output)

        # Compute the final cost: output.T * (Q * output)
        cost = torch.matmul(output.T, Q_output)
        return cost
