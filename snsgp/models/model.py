import torch
import gc
import numpy as np
import torch.autograd.profiler as profiler

class SNSGP(torch.nn.Module):
    def __init__(self, X, y, X_bar=None, Xm=None,
                 jitter=10**-8, random_state=None, local_noise=True, local_std=True, 
                 device='cuda', debug=False):
        super().__init__()

        assert len(
            X.shape) == 2, "X is expected to have shape (n, m) but has shape "+str(X.shape)
        assert len(
            y.shape) == 2, "y is expected to have shape (n, 1) but has shape "+str(y.shape)
        assert y.shape[1] == 1, "y is expected to have shape (n, 1) but has shape "+str(
            y.shape)

        self.X = X
        self.raw_mean = y.mean()
        self.y = y - self.raw_mean
        self.X_bar = X_bar
        self.Xm = Xm
        self.debug = debug
        self.N = self.X.shape[0]
        self.m = self.Xm.shape[0]
        self.input_dim = self.X.shape[1]

        self.num_latent_points = self.X_bar.shape[0]

        self.jitter = jitter
        self.local_noise = local_noise
        self.local_std = local_std
        self.random_state = random_state

        # Local params
        self.local_gp_ls = self.param((self.input_dim,))
        self.local_gp_std = self.param((self.input_dim,))
        if not self.local_std:
            self.local_gp_std.requires_grad = False
        self.local_gp_noise_std = self.param((self.input_dim,))
        if not self.local_noise:
            self.local_gp_noise_std.requires_grad = False
        self.local_ls = self.param(
            (self.num_latent_points, self.input_dim))

        # Global params
        self.global_gp_std = self.param((1,))
        self.global_gp_noise_std = self.param((1,))

        # Other params to be used
        # self.eye_num_latent_points = torch.eye(self.num_latent_points, dtype=self.X.dtype)
        # self.eye_N = torch.eye(self.N)
        self.pi = torch.tensor(np.pi)

        # Initialize model parameters
        self.initialize_params()

    def param(self, shape, requires_grad=True):
        return torch.nn.Parameter(torch.empty(shape, dtype=self.X.dtype), requires_grad=requires_grad)

    def initialize_params(self):
        if self.random_state is None:
            self.random_state = int(torch.rand(1)*1000)
        torch.manual_seed(self.random_state)
        for param in self.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0.0, std=1.0)
            else:
                torch.nn.init.constant_(param, 1.)

    def LocalKernel(self, x1, x2, dim):  # kernel of local gp (GP_l)
        dist = torch.square(x1 - x2.T)
        dist[dist==0] = 10**-20
        scaled_dist = dist/self.local_gp_ls[dim]**2
        # print(scaled_dist, self.local_gp_ls[dim]**2)
        return self.local_gp_std[dim]**2 * torch.exp(-0.5*scaled_dist)

    def get_LS(self, X, dim, train=False):  # Infer lengthscales for train_X (self.X)
        # print('Log: dim', dim)
        k = self.LocalKernel(
            self.X_bar[:, dim, None], self.X_bar[:, dim, None], dim)

        # Diagonal Solution from https://stackoverflow.com/a/48170846/13330701
        dk = k.diagonal()
        dk += self.local_gp_noise_std[dim]**2
        c = torch.linalg.cholesky(k)
        alpha = torch.cholesky_solve(
            torch.log(torch.abs(self.local_ls[:, dim, None])), c)
        k_star = self.LocalKernel(
            X[:, dim, None], self.X_bar[:, dim, None], dim)
        l = torch.exp(k_star@alpha)

        if train:
            k_star = self.LocalKernel(
                X[:, dim, None], self.X_bar[:, dim, None], dim)
            k_star_star = self.LocalKernel(
                X[:, dim, None], X[:, dim, None], dim)

            chol = torch.linalg.cholesky(k)
            v = torch.cholesky_solve(k_star.T, chol)

            k_post = k_star_star - k_star@v
            k_post_det = torch.det(k_post)
            k_post_det = torch.clamp(k_post_det, min=10**-20)
            # if k_post_det<=0:
            #     k_post_det = torch.tensor(10**-20)
            B = torch.log(k_post_det).reshape(-1,1)
            # dk_post = k_post.diagonal()
            # dk_post += self.jitter
            # post_chol = torch.linalg.cholesky(k_post)
            # B.append(torch.log(post_chol.diagonal()))
            return l, B
        else:
            return l

    def GlobalKernel(self, X1, X2, train=False):  # global GP (GP_y)
        suffix = None
        scaled_dist = None
        B_all = None
        for d in range(X1.shape[1]):
            if self.debug:
                input('start:')
            if train:
                l, B = self.get_LS(X1, d, train)
                if B_all is None:
                    B_all = B
                else:
                    B_all +=  B
                if self.debug:
                    input('0:')
                l1 = l
                l2 = l
            else:
                l1 = self.get_LS(X1, d)
                l2 = self.get_LS(X2, d)
            if self.debug:
                input('1:')
            lsq = torch.square(l1) + torch.square(l2.T)
            if suffix is None:
                suffix = torch.sqrt(2 * l1@l2.T / lsq)
            else:
                suffix = suffix * torch.sqrt(2 * l1@l2.T / lsq)
            if self.debug:    
                input('2:')
            dist = torch.square(X1[:, None, d] - X2[None, :, d])
            if scaled_dist is None:
                scaled_dist = dist/lsq
            else:
                scaled_dist = scaled_dist + dist/lsq
            if self.debug:
                input('3:')

            # del l, l1, l2, B, lsq, dist
            # gc.collect()
            # torch.cuda.empty_cache()

        K = self.global_gp_std**2 * \
            suffix * torch.exp(-scaled_dist)

        if train:
            return K, B
        else:
            return K

    def add_jitter(self, mat):
        diag = mat.diagonal()
        diag += self.jitter

    def forward(self):
        self.K_mm, Bm = self.GlobalKernel(self.Xm, self.Xm, train=True)
        self.add_jitter(self.K_mm)

        self.K_mn = self.GlobalKernel(self.Xm, self.X)

        self.K_inner = self.K_mm + self.global_gp_noise_std**-2 * self.K_mn@self.K_mn.T
        self.add_jitter(self.K_inner)
        # print('ranks', torch.linalg.matrix_rank(self.K_inner))
        # self.L = torch.linalg.cholesky(self.K_inner)
        # v = torch.cholesky_solve(self.K_mn, self.L)
        # self.K_tilde_inv = -(self.K_mn.T@v)/(self.global_gp_noise_std**4)
        self.K_tilde_inv = -(self.K_mn.T@torch.linalg.inv(self.K_inner)@self.K_mn)/(self.global_gp_noise_std**4)
        diag = self.K_tilde_inv.diagonal()
        diag += self.global_gp_noise_std**-2

        # Using https://en.wikipedia.org/wiki/Matrix_determinant_lemma
        d1 = self.X.shape[0] * torch.log(self.global_gp_noise_std**2)
        d2 = torch.sum(torch.log(torch.linalg.cholesky(self.K_mm).diagonal()))
        # d3 = torch.sum(torch.log(self.L.diagonal()))
        d3 = torch.log(torch.clamp(torch.det(self.K_inner), min=10**-20))
        # print('d123', d1.item(), d2.item(), d3.item())
        self.K_tilde_logdet =  d1 + d3 - d2

        return 0.5 * (self.y.T@self.K_tilde_inv@self.y + self.K_tilde_logdet + self.y.shape[0] * 2 * self.pi)/self.X.nelement()

        # dK = K.diagonal()
        # dK += self.global_gp_noise_std**2 + self.jitter
        # # print(K)
        # L = torch.linalg.cholesky(K)
        # alpha = torch.cholesky_solve(self.y, L)
        
        # Apart1 = self.y.T@alpha
        # Apart2 = torch.sum(torch.log(L.diagonal()))
        # Apart2 = torch.det(K)
        # print('Before Apart2', Apart2)
        # Apart2 = Apart2.clamp(Apart2, min=10**-20)
        # Apart2 = torch.log(Apart2)
        # Apart3 = self.N * torch.log(2*self.pi)

        # A = 0.5*( Apart1 + Apart2)[0, 0]
        
        # Bpart1 = B
        # Bpart2 = 0.5*(self.num_latent_points *
        #                                    self.input_dim*torch.log(2*self.pi))
        
        # B = Bpart1# + Bpart2
        
        # print("A1", Apart1, "A2", Apart2, "B", B, "Loss", A+B, 'local var', self.local_gp_std)
        # return (A+Bm)/self.N/self.input_dim

    def predict(self, X_new):  # Predict at new locations
        # K = self.GlobalKernel(self.X, self.X)
        K_star = self.GlobalKernel(X_new, self.X)
        K_star_star = self.GlobalKernel(X_new, X_new)

        # dK = K.diagonal()
        # dK += self.global_gp_noise_std**2
        # L = torch.linalg.cholesky(K)
        # alpha = torch.cholesky_solve(self.y, L)

        self.K_mm = self.GlobalKernel(self.Xm, self.Xm)
        self.add_jitter(self.K_mm)

        self.K_mn = self.GlobalKernel(self.Xm, self.X)

        self.K_inner = self.K_mm + self.global_gp_noise_std**-2 * (self.K_mn@self.K_mn.T)
        self.add_jitter(self.K_inner)
        # self.L = torch.linalg.cholesky(self.K_inner)
        
        # v = torch.cholesky_solve(self.K_mn, self.L)
        self.K_tilde_inv = -(self.K_mn.T@torch.linalg.inv(self.K_inner)@self.K_mn)/(self.global_gp_noise_std**4)
        diag = self.K_tilde_inv.diagonal()
        diag += self.global_gp_noise_std**-2
                
        pred_mean = K_star@self.K_tilde_inv@self.y + self.raw_mean

        # v = torch.cholesky_solve(K_star.T, L)
        pred_var = K_star_star - K_star@self.K_tilde_inv@K_star.T

        dpred_var = pred_var.diagonal()
        dpred_var += self.global_gp_noise_std**2
        return pred_mean, pred_var
