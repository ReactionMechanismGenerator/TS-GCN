import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
import torch_geometric as tg
from torch_geometric.nn import GCNConv

# used to access the updated GCN later
from model.GNN import GNN, MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # self.gcn = GCNConv(**initialization_params)
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
        # or use the MLP
        # x = MLP(x, num_layers, out_dim)

        # x = dist_nlsq(D, W, )
        return x


class G2C(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=300, depth=3, n_layers=2):
        super(G2C, self).__init__()
        self.gnn = GNN(node_dim, edge_dim, hidden_dim, depth, n_layers)
        self.edge_mlp = MLP(hidden_dim, hidden_dim, n_layers)
        self.pred = Linear(hidden_dim, 2)
        self.act = torch.nn.Softplus()

    def forward(self, data):
        # torch.autograd.set_detect_anomaly(True)   # use only when debugging
        # get updated edge attributes. edge_attr: [E, F_e]
        _, edge_attr = self.gnn(data.x, data.edge_index, data.edge_attr)
        edge_embed = self.edge_mlp(edge_attr)
        # make prediction of distance matrix and weight matrix
        edge_pred = self.pred(edge_embed)   # shape: E x 2
        edge_pred = tg.utils.to_dense_adj(data.edge_index, data.batch, edge_pred)   # shape: b x N x N x 2

        # mask
        diag_mask = tg.utils.to_dense_adj(data.edge_index, data.batch)  # diagonals are masked too i.e. 0s along diagonal
        edge_pred = edge_pred + edge_pred.permute([0, 2, 1, 3])

        # 0s the diagonal and any extra rows/cols for smaller molecules with n < N_max
        preds = self.act(edge_pred) * diag_mask.unsqueeze(-1)
        D, W = preds.split(1, dim=-1)

        N_fill = torch.cat([torch.arange(x) for x in data.batch.bincount()])
        mask = diag_mask.clone()
        mask[data.batch, N_fill, N_fill] = 1  # fill diagonals

        X = self.dist_nlsq(D.squeeze(-1), W.squeeze(-1), mask)
        data.coords = X

        return diag_mask*self.distances(X), diag_mask


    def distance_to_gram(self, D, mask):
        """Convert distance matrix to gram matrix"""
        # D shape is (batch, 21, 21)
        # mask is (batch, 21, 21)
        # N_f32 is (batch,)
        # N_f32, [-1,1,1]) is (batch, 1, 1)
        Nf32 = mask.nonzero().sum()  # number of atoms in each batch
        D = torch.square(D)
        D_row = torch.sum(D, dim=1, keepdim=True)
        D_col = torch.sum(D, dim=2, keepdim=True)
        D_mean = torch.sum(D, dim=[1,2], keepdim=True)
        G = mask * -0.5 * (D - D_row - D_col + D_mean)
        return G

    def low_rank_approx_power(self, A, k=3, num_steps=10):
        A_lr = A    # A shape (batch, 21, 21)
        u_set = []
        for kx in range(k):
            # initialize eigenvector. u shape (1, 21, 1)
            u = torch.unsqueeze(torch.normal(mean=0, std=1, size=A.shape[:-1]), dim=-1).to(device)
            # power iteration
            for j in range(num_steps):
                u = F.normalize(u, dim=1, p=2, eps=1e-3)
                u = torch.matmul(A_lr, u)
            # rescale by scalar value sqrt(eigenvalue)
            eig_sq = torch.sum(torch.square(u), dim=1, keepdim=True)    # eig_sq shape (1,1,1)
            # normalization step
            u = u/ torch.pow(eig_sq + 1e-2, 0.25)   # u shape (batch, 21, 1)
            u_set.append(u)
            # transpose columns 1 and 2 so that torch.matmul(u, u.transpose(1,2)) has shape (batch, 21, 21)
            A_lr = A_lr - torch.matmul(u, u.transpose(1,2)) # A_lr shape (batch, 21, 21)

        X = torch.cat(tensors=u_set, dim=2)         # X shape (1, 21, 3)
        return X

    def dist_nlsq(self, D, W, mask):
        """
        Solve a nonlinear distance geometry problem by nonlinear least squares

        Objective is Sum_ij w_ij (D_ij - |x_i - x_j|)^2
        """
        # D is (batch, 21, 21)
        # W is (batch, 21, 21)
        # mask is (batch, 21, 21)

        T = 100
        eps = 0.1
        alpha = 5.0
        alpha_base = 0.1

        def gradfun(X):
            """ Grad function """
            # X is (batch, 21, 3)
            # must make X a variable to use autograd
            X = Variable(X, requires_grad=True)

            D_X = self.distances(X)  # D_X is (batch, 21, 21)

            # Energy calculation
            U = torch.sum(mask * W * torch.square(D - D_X), dim=[1, 2]) / torch.sum(mask, dim=[1, 2])
            U = torch.sum(U)  # U is a scalar

            # Gradient calculation
            # U = Variable(U, requires_grad=True)
            g = torch.autograd.grad(U, X)[0]
            return g


        def stepfun(t, x_t):
            """Step function"""
            # x_t is (?, 21, 3)
            g = gradfun(x_t)
            dx = -eps * g  # (?, 21, 3)

            # Speed clipping (How fast in Angstroms)
            speed = torch.sqrt(torch.sum(torch.square(dx), dim=2, keepdim=True) + 1E-3) # (batch, 21, 3)

            # Alpha sets max speed (soft trust region)
            alpha_t = alpha_base + (alpha - alpha_base) * torch.tensor((T - t) / T).float()
            scale = alpha_t * torch.tanh(speed / alpha_t) / speed  # (batch, 21, 1)
            dx *= scale  # (batch, 21, 3)

            x_new = x_t + dx

            return t + 1, x_new

        # intial guess for X
        # D is (batch, 21, 21)
        # mask is (batch, 21, 21)
        B = self.distance_to_gram(D, mask)       # B is (batch, 21, 21)
        x_init = self.low_rank_approx_power(B)   # x_init is (batch, 21, 3)

        # prepare simulation
        max_size = D.size(1)
        x_init += torch.normal(mean=0, std=1, size=[D.shape[0], max_size, 3]).to(device)

        # Optimization loop
        t=0
        x = x_init
        while t < T:
           t, x = stepfun(t, x)

        return x

    # currently not used
    def rmsd(self, X1, X2, mask_V):
        """RMSD between 2 structures"""
        # note: in torch, sum and mean automatically reduce dimensions
        X1 = X1 - torch.sum(mask_V * X1,dim=1,keepdim=True) \
                    / torch.sum(mask_V, dim=1,keepdim=True)
        X2 = X2 - torch.sum(mask_V * X2, dim=1, keepdim=True) \
             / torch.sum(mask_V, dim=1, keepdim=True)

        X1 *= mask_V
        X2 *= mask_V

        eps = 1E-2

        X1_perturb = X1 + eps * torch.normal(mean=0, std=1, size=X1.shape)
        # or use random normal ~N(0, 1)
        X2_perturb = X2 + eps * torch.randn(X2.shape)
        A = torch.matmul(X1_perturb.T, X2_perturb)
        S, U, V = torch.svd(A)  # default arguments are: some=True, compute_uv=True

        X1_align = torch.matmul(U, torch.matmul(V, X1.T))
        X1_align = X1_align.permute(0, 2, 1)

        MSD = torch.sum(mask_V * torch.square(X1_align - X2), [1, 2]) \
              / torch.sum(mask_V, [1, 2])
        RMSD = torch.mean(torch.sqrt(MSD + 1E-3))
        return RMSD, X1_align

    def distances(self, X):
        """Compute Euclidean distance from X"""
        # X is (batch, 21, 3)
        # D is (batch, 21, 21)

        Dsq = torch.square(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2))
        D = torch.sqrt(torch.sum(Dsq, dim=3) + 1E-2)
        return D
