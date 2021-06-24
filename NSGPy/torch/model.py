import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from sklearn.cluster import KMeans

class NSGPRegression():
    def __init__(self, X, y, num_inducing_points, f_indu, params, seed=0):
        self.num_inducing_points = num_inducing_points
        self.X = X
        self.y = y
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.params = params

        assert len(X.shape) == 2
        assert len(y.shape) == 2
        self.input_dim = X.shape[1]
        
        # Defining X_bar (Locations where latent lengthscales are to be learnt)
        # XY_choice = tf.concat([self.X, self.y], dim=1)
        self.X_bar = torch.tensor(f_indu(self.X, num_inducing_points))
        
#         self.X_bar = X[np.random.choice(X.shape[0], self.num_inducing_points, replace=False)]
        
        # initialize params
        # self.init(seed)
        
    def init(self, seed):
        torch.manual_seed(seed)
        f = lambda size: (1+torch.rand(size)).requires_grad_()
        self.params['local_gp_std'] = f((self.input_dim,))
        self.params['local_gp_ls'] = f((self.input_dim,))
        self.params['local_gp_noise_std_std'] = f((self.input_dim,))
        self.params['local_ls'] = f((self.num_inducing_points, self.input_dim))
        self.params['global_gp_std'] = f((1,))
        self.params['global_gp_noise_std_std'] = f((1,))
        
#         self.dparams = self.params.copy() # Saving gradients
        
    def LocalKernel(self,x1,x2,dim): # return local kernel without variance
        x1 = x1.reshape(-1,1)
        x2 = x2.reshape(-1,1)
        
        dist = x1 - x2.T
        scaled_dist = dist/self.params['local_gp_ls'][dim]
        
        return torch.exp(-torch.square(scaled_dist)/2)
    
    def get_LS(self, X): # Getting lengthscales for entire train_X (self.X)
        l_list = []
        for dim in range(self.input_dim):
            k = self.params['local_gp_std'][dim]**2 * self.LocalKernel(self.X_bar[:,dim], self.X_bar[:,dim], dim)
            k = k + (torch.eye(k.shape[0])*self.params['local_gp_noise_std'][dim]**2)
            c = torch.linalg.cholesky(k)
#             print(c.dtype, self.params['local_ls'][:, dim:dim+1].dtype)
#             print(self.params['local_ls'][:, dim:dim+1].shape)
            alpha = torch.cholesky_solve(self.params['local_ls'][dim, :].view(-1,1), c)
#             print(alpha)
            
            k_star = self.params['local_gp_std'][dim]**2 * self.LocalKernel(X[:,dim], self.X_bar[:,dim], dim)
            l = k_star@alpha
            l_list.append(l)
        
        return l_list
    
    def GlobalKernel(self, X1, X2): # Construct global GP
        l1 = torch.cat(self.get_LS(X1), dim=1)[:,None,:]
        l2 = torch.cat(self.get_LS(X2), dim=1)[None,:,:]
        lsq = torch.square(l1) + torch.square(l2)
        suffix = torch.sqrt(2 * l1 * l2 / lsq).prod(axis=2) # (n, m)
        ##############################################
#         print((X1[:,None,:] - X2[None,:,:]).shape, l1l2b2sqr.shape)
        scaled_dist = torch.square(X1[:,None,:] - X2[None,:,:])/lsq # (n, m, d)
        K = suffix * torch.exp(-scaled_dist.sum(dim=2))
        return K
        
    def nlml(self, X, y):
        B = []
        for dim in range(self.input_dim):
            k = self.LocalKernel(self.X_bar[:,dim], self.X_bar[:,dim], dim)
            k = k + (torch.eye(k.shape[0])*self.params['local_gp_noise_std'][dim]**2)
            
            c = torch.linalg.cholesky(k)
            B.append(torch.log(c.diagonal()))
        
        B = torch.sum(torch.cat(B))
        
        K = self.params['global_gp_std']**2 * self.GlobalKernel(X, X)
        K = K + (torch.eye(K.shape[0])*self.params['global_gp_noise_std']**2)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y, L)
        A = 0.5*(y.T@alpha + torch.sum(torch.log(L.diagonal())))[0,0]
        return A+B
    
    def gradient(self, A_inv, B_inv, dA, dB): # In progress
        first = -self.y.T@A_inv@dA@A_inv@self.y
        second = np.sum(np.diag(A_inv@dA))
        third = np.sum(np.diag(B_inv@dB))
    
    def optimize_manual(self, lr=0.01, epochs=100, store_history=False): # In progress
        loss = []
        for epoch in range(epochs):
            # Some fixed vals
            A = self.params['global_gp_std']**2 * self.GlobalKernel(self.X, self.X)
            A_inv = np.linalg.inv(A)
            
            # Calculating gradients
            zA = np.zeros(A.shape)
            zB = np.zeros((self.num_inducing_points, 
                           self.num_inducing_points))
            for dim in range(self.input_dim):
                x1 = self.X_bar[:,dim][:, None]
                B = self.params['local_gp_std']**2 * self.LocalKernel(x1)
                B_inv = np.linalg.inv(B)
                
                self.dparams['local_gp_std'][dim] = self.gradient(A_inv, B_inv, 2*self.params['local_gp_std'][dim] *\
                                                    self.LocalKernel(x1, x1), zB)
                self.dparams['local_gp_ls'][dim] = self.params['local_gp_std'][dim]**2 *\
                                                    self.LocalKernel(x1, x1) *\
                                                    np.square(x1-x1.T)/(self.params['local_gp_ls']**3)
                self.dparams['local_gp_noise_std'][dim] = 0
                                                    
            if store_history:
                loss.append(self.nlml(self.X, self.y))
                
    def optimize_auto(self, epochs=10, lr=0.01, gran=10, m=0, optim='sgd'):
        def closure():
            optim.zero_grad()
            loss = self.nlml(self.X, self.y)
            loss.backward()
            with torch.no_grad():
                for p, param in self.params.items():
                    param.clamp_(10**-20, np.inf)
            return loss
                
        if optim == 'sgd':
            optim = torch.optim.SGD(self.params.values(), lr=lr, momentum=m)
            for epoch in range(epochs):
                loss = closure()
                if epoch%gran==0:
                    print(loss.item())
                optim.step()
        elif optim == 'adam':
            optim = torch.optim.Adam(self.params.values(), lr=lr)
            for epoch in range(epochs):
                loss = closure()
                if epoch%gran==0:
                    print(loss.item())
                optim.step()
        elif optim == 'lbfgs':
            optim = torch.optim.LBFGS(self.params.values(), lr=lr, max_iter=epochs)
            optim.step(closure)
        
    def predict(self, X_new): # Predict at new locations
        
        K = self.params['global_gp_std']**2 * self.GlobalKernel(self.X, self.X)
        K_star = self.params['global_gp_std']**2 * self.GlobalKernel(X_new, self.X)
        K_star_star = self.params['global_gp_std']**2 * self.GlobalKernel(X_new, X_new)
        
        L = torch.linalg.cholesky(K + torch.eye(self.X.shape[0]) * self.params['global_gp_noise_std']**2)
        alpha = torch.cholesky_solve(self.y, L)
        
        pred_mean = K_star@alpha
        
        v = torch.cholesky_solve(K_star.T, L)
        pred_var = K_star_star + torch.eye(X_new.shape[0])*self.params['global_gp_noise_std']**2 - K_star@v
        
        return pred_mean, pred_var