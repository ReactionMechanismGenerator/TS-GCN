import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
