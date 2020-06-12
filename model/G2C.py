import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv

# used to access the updated GCN later
import GNN


# define dense net
class DenseNet(torch.nn.Module):
    def __init__(self, hidden_nodes, output_size):
        super(DenseNet, self).__init__()
        self.fc1 = Linear(hidden_nodes, hidden_nodes)
        self.bn1 = BatchNorm1d(hidden_nodes)
        self.fc2 = Linear(hidden_nodes, hidden_nodes)
        self.bn2 = BatchNorm1d(hidden_nodes)
        self.fc3 = Linear(hidden_nodes, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = F.dropout(x, p=0.2, training=self.training)

        x = self.fc3(x)
        x = F.relu(x)

        return x


class StandardGCN(torch.nn.Module):
    """
    This class will be replaced by the custom GNN class in GNN.py
    GCNConv only utilizes atom features, while we want to utilize both atom and bond features, so GNN will incorporate this.
    """
    def __init__(self, input_features, hidden_nodes):
        super(StandardGCN, self).__init__()

        # do graph convolution 3x and batch normalize after each
        # if define cache=True, the shape of batch must be same!
        self.conv1 = GCNConv(input_features, hidden_nodes, cached=False)
        self.bn1 = BatchNorm1d(hidden_nodes)

        self.conv2 = GCNConv(hidden_nodes, hidden_nodes, cached=False)
        self.bn2 = BatchNorm1d(hidden_nodes)

        self.conv3 = GCNConv(hidden_nodes, hidden_nodes, cached=False)
        self.bn3 = BatchNorm1d(hidden_nodes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)

        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)

        return x


class CombinedNN(torch.nn.Module):
    """
    NN architecture that consists of 2 parts:
        1) Custom GNN that builds off of GCNConv from pytorch geometric.
           GCNConv only utilizes atom features, while we want to utilize both atom and bond features.
           Performs 3 iterations of graph convolution
        2) FC network that outputs prediction of TS distance matrix D_init
    """
    def __init__(self):
        super(CombinedNN, self).__init__()
        # do graph convolution twice and batch normalize after each
        # if you defined cache=True, the shape of batch must be same!

        #todo: Assumes that GNN defines 3 layers with batch normalization, ReLU activation, and any global_add_pool
        # initialization_params = {'iterations':3,
        #                          'input_features':70,
        #                          'hidden_layers':100
        #                          # etc. add other inputs here
        #                          }
        # self.gcn = GNN.GCNConv(**initialization_params)
        initialization_params = {'input_features': 75,
                                 'hidden_nodes': 100
                                 }
        self.gcn = StandardGCN(**initialization_params)

        # could also use GNN.Module1 to create MLP with customizable number of layers
        initialization_params = {'hidden_nodes': 100,
                                     'output_size': 3
                                 }
        self.fc = DenseNet(**initialization_params)

    def forward(self, data):
        """
        data is an instance of the pytorch geometric Data class. Example below
        data = Data(x=torch.tensor(node_f, dtype=torch.float),
              edge_index=torch.tensor(edge_index, dtype=torch.long),
              edge_attr=torch.tensor(edge_attr,dtype=torch.float)
              )
        """
        x = self.gcn(data)
        x = self.fc(x)
        return x
