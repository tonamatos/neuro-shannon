from pyqubo import Array
import numpy as np
import torch


class QUBO:
    def __init__(self, p1, p2) -> None:
        super(QUBO, self).__init__()
        self.p1 = p1
        self.p2 = p2

    def create_mis_model(self, edges, num_nodes):
        X = Array.create("X", shape=(num_nodes,), vartype="BINARY")

        hamiltonian = -self.p1 * sum(X)
        print(edges[0], edges[1])
        for i in range(len(edges[0])):
            u = edges[0][i].item()
            v = edges[1][i].item()

            hamiltonian += self.p2*(X[u] * X[v])
        
        return hamiltonian.compile()
    
    def create_Q_matrix(self, edges, num_nodes):
        model = self.create_mis_model(edges, num_nodes)

        extract_val = lambda x: int(x[2:-1])
        Q_matrix = np.zeros((num_nodes, num_nodes))

        qubo_dict, _ = model.to_qubo()

        for (a, b), quv in qubo_dict.items():
            u = min(extract_val(a), extract_val(b))
            v = max(extract_val(a), extract_val(b))
            Q_matrix[u, v] = quv

        return torch.tensor(Q_matrix, dtype=torch.float32)
    
    def qubo_approx_cost(self, output, Q):
        cost = torch.sum(torch.matmul(torch.matmul(output.T, Q), output))
        return cost