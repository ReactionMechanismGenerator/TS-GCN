import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric as tg

# used to access the updated GCN later
from model.GNN import GNN, MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class G2C(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=300, depth=3, n_layers=2):
        super(G2C, self).__init__()
        self.gnn = GNN(node_dim, edge_dim, hidden_dim, depth, n_layers)
        self.edge_mlp = MLP(hidden_dim, hidden_dim, n_layers)
        self.pred = Linear(hidden_dim, 2)
        self.act = torch.nn.Softplus()
        self.d_init = torch.nn.Parameter(torch.tensor([4.]), requires_grad=True).to(device)

        # learnable optimization params
        self.T = torch.nn.Parameter(torch.tensor([50.]), requires_grad=True).to(device)
        self.eps = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True).to(device)
        self.alpha = torch.nn.Parameter(torch.tensor([5.]), requires_grad=True).to(device)
        self.alpha_base = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True).to(device)

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
        preds = self.act(self.d_init + edge_pred) * diag_mask.unsqueeze(-1)
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
        N_mol = mask.sum(dim=1)[:, 0].view(-1, 1, 1)  # number of atoms per mol
        D = torch.square(D)
        D_row = torch.sum(D, dim=1, keepdim=True) / N_mol
        D_col = torch.sum(D, dim=2, keepdim=True) / N_mol
        D_mean = torch.sum(D, dim=[1,2], keepdim=True) / torch.square(N_mol)
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

    def dist_nlsq(self, D, W, C, Wc, mask, tetra_mask):
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
            dx = -self.eps * g  # (?, 21, 3)

            # Speed clipping (How fast in Angstroms)
            speed = torch.sqrt(torch.sum(torch.square(dx), dim=2, keepdim=True) + 1E-3) # (batch, 21, 3)

            # Alpha sets max speed (soft trust region)
            alpha_t = self.alpha_base + (self.alpha - self.alpha_base) * ((self.T - t) / self.T)
            scale = alpha_t * torch.tanh(speed / alpha_t) / speed  # (batch, 21, 1)
            dx_scaled = dx * scale  # (batch, 21, 3)

            x_new = x_t + dx_scaled

            return t + 1, x_new

        # initial guess for X
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

    def distances(self, X):
        """Compute Euclidean distance from X"""
        # X is (batch, 21, 3)
        # D is (batch, 21, 21)

        Dsq = torch.square(torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2))
        D = torch.sqrt(torch.sum(Dsq, dim=3) + 1E-2)
        return D
