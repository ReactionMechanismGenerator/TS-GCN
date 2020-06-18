import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_sum
from torch_geometric.nn import MetaLayer


def MLP(in_dim, num_layers, out_dim, activation=torch.nn.ReLU()):
    """
    Create a variable layer perceptron algorithm with default ReLU activation.

    Inputs:
        input_h (np array):         1D np array to pass through the network.
        num_layers (int):           number of layers in the network.
        out_dim (int):              number of features input and output from each hidden layer, 
                                    including the output layer.
        activation (torch function): activation function to be used during the hidden layers
    """
    model = Module1(in_dim, out_dim, num_layers, activation=activation)

    h = model(torch.from_numpy(input_h).float())

    return h


class Module1(nn.Module):
    """
    Creates a NN using nn.ModuleList to automatically adjust the number of layers.
    For each hidden layer, the number of inputs and outputs is constant.

    Inputs:
        in_dim (int):               number of features contained in the input layer.
        out_dim (int):              number of features input and output from each hidden layer, 
                                    including the output layer.
        num_layers (int):           number of layers in the network
        activation (torch function): activation function to be used during the hidden layers
    """
    def __init__(self, in_dim, out_dim, num_layers, activation=torch.nn.ReLU()):
        super(Module1, self).__init__()
        self.layers = nn.ModuleList()
        # create the input layer
        self.layers.append(nn.Linear(in_dim, out_dim))
        self.layers.append(activation)

        # create hidden layers
        if num_layers > 2:
            for i in range(num_layers - 2):
                # print(out_dim)
                # print(type(out_dim))
                self.layers.append(nn.Linear(out_dim, out_dim))
                self.layers.append(activation)
        
        # create output layer
        # self.layers.append(nn.Linear(out_dim, out_dim))  # linear output
        self.layers.append(torch.nn.Sigmoid())    # sigmoid output
        # self.layers.append(torch.nn.Tanh())       # tanh output
        
    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        return y


# test the MLP
mlp = Module1(4,3,6)
print(mlp)
mlp(torch.from_numpy(np.array([4,5,6,7])).float())


# test the MLP using the function that calls the pytorch class
input_h = np.array([1, 2, 3, 4, 5])
h = MLP(input_h, 4, 4)
print(h)


class Module2(nn.Module):
    """
    Creates a NN using nn.ModuleList to automatically adjust the number of layers.
    Allows the user to customize the number of nodes at each hidden layer.

    Inputs:
        h_sizes (list):             number of features input to each layer. 
                                    length of h_sizes will determine the number of layesr.
        out_dim (int):              number of features in the final output from the network.
        activation (torch function): activation function to be used during the hidden layers

    """
    def __init__(self, h_sizes, out_dim, activation=torch.nn.ReLU()):
        super(Module2, self).__init__()
        self.layers = nn.ModuleList()
        num_layers = len(h_sizes)
        
        # create hidden layers
        for k in range(num_layers - 1):
            self.layers.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.layers.append(activation)
        
        # create output layer
        # could also just have out_dim be the last entry in h_sizes
        self.layers.append(nn.Linear(h_sizes[-1], out_dim))
        # self.layers.append(nn.Linear(dIn, dOut))  # linear output
        self.layers.append(torch.nn.Sigmoid())    # sigmoid output
        # self.layers.append(torch.nn.Tanh())       # tanh output
        
    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        return y

# test the MLP
mlp = Module2([2, 5, 55, 6, 6, 7], 5)
print(mlp)
print(mlp(torch.from_numpy(np.array([4,5])).float()))


class EdgeModel(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super(EdgeModel, self).__init__()
        self.edge = Lin(hidden_dim, hidden_dim)
        self.node_in = Lin(hidden_dim, hidden_dim)
        self.node_out = Lin(hidden_dim, hidden_dim)
        self.mlp = MLP(hidden_dim, n_layers, hidden_dim)

    def forward(self, src, tgt, edge_attr, u, batch):
        # source, target: [E, h], where E is the number of edges.
        # edge_attr: [E, h]
        # u: [B, h], where B is the number of graphs (we don't have any of these yet)
        # batch: [E] with max entry B - 1.

        f_ij = self.edge(edge_attr)
        f_i = self.node_in(src)
        f_j = self.node_out(tgt)

        out = F.relu(f_ij + f_i + f_j)
        return self.mlp(out)


class NodeModel(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = MLP(hidden_dim, n_layers, hidden_dim)
        self.node_mlp_2 = MLP(hidden_dim, n_layers, hidden_dim)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, h], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, h]
        # u: [B, F_u] (N/A)
        # batch: [N] with max entry B - 1.
        _, col = edge_index
        out = self.node_mlp_1(edge_attr)
        out = scatter_sum(out, col, dim=0, dim_size=x.size(0))
        return self.node_mlp_2(out)


class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=300, depth=3, n_layers=2):
        super(GNN, self).__init__()
        self.depth = depth
        self.node_init = Lin(node_dim, hidden_dim)
        self.edge_init = Lin(edge_dim, hidden_dim)
        self.update = MetaLayer(EdgeModel(hidden_dim, n_layers), NodeModel(hidden_dim, n_layers))

    def forward(self, x, edge_index, edge_attr):

        x = self.node_init(x)
        edge_attr = self.edge_init(edge_attr)
        for _ in range(self.depth):
            x, edge_attr, _ = self.update(x, edge_index, edge_attr)
        return x, edge_attr
