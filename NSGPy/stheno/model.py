# Common imports
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np

# Stheno related imports
import lab.tensorflow as B
from stheno import GP, EQ
from varz.tensorflow import Vars, minimise_l_bfgs_b

# Import NSEQ kernel
from .nseq import NSEQ

class NSGPRegression:
    def __init__(self, X, y, vs, num_inducing_points, f_indu, seed=0):
        self.num_inducing_points = num_inducing_points
        self.X = X
        self.y = y
        self.vs = vs

        assert len(X.shape) == 2
        assert len(y.shape) == 2
        self.input_dim = X.shape[1]
        B.random.set_random_seed(seed)
        self.X_bar = f_indu(self.X, num_inducing_points) # f to select inducing points
        
        # Initialize hyperparams
        # self.init_params(seed)
        
    def init_params(self, seed):
        np.random.seed(seed)
        rand = lambda shape: np.abs(np.random.normal(loc=0, scale=1, size=shape))
        
        # Local params
        self.vs.positive(init=rand((self.input_dim,)), shape=(self.input_dim,), name='local_gp_std')
        self.vs.positive(init=rand((self.input_dim,)), shape=(self.input_dim,), name='local_gp_ls')
        self.vs.positive(init=rand((self.input_dim, self.num_inducing_points)), 
                            shape=(self.input_dim, self.num_inducing_points), name='local_ls')
        self.vs.positive(init=rand((self.input_dim,)), shape=(self.input_dim,), name='local_gp_noise_std')

        # Global params
        self.vs.positive(init=np.random.normal(), name='global_gp_std')
        self.vs.positive(init=np.random.normal(), name='global_gp_noise_std')

    def LocalGP(self, vs, X, return_logpdf=False): # Getting lengthscales for entire train_X (self.X)
        l_list = []
        if return_logpdf:
            log_pdf_list = []
        for dim in range(self.input_dim):
            f = GP(vs['local_gp_std'][dim]**2 * EQ().stretch(vs['local_gp_ls'][dim]))
            f_post = f | (f(self.X_bar[:, dim], vs['local_gp_noise_std'][dim]**2), vs['local_ls'][dim])
            l = f_post(X[:, dim]).mean.mat
            l_list.append(l)
            
            if return_logpdf:
                log_pdf_list.append(f(self.X_bar[:, dim], vs['local_gp_noise_std'][dim]**2).logpdf(vs['local_ls'][dim]))
        
        if return_logpdf:
            return l_list, log_pdf_list
        else:
            return l_list
    
    def GlobalGP(self, vs): # Construct global GP and return nlml
        l_list, log_pdf_list = self.LocalGP(vs, self.X, return_logpdf=True)
        global_ls = tf.concat(l_list, axis=1)
        
        f = GP(vs['global_gp_std']**2 * NSEQ(global_ls, global_ls))
        
        return -f(self.X, vs['global_gp_noise_std']**2).logpdf(self.y) - tf.reduce_sum(log_pdf_list)
    
    def optimize(self, iters=1000, jit=False, trace=False): # Optimize hyperparams
        minimise_l_bfgs_b(self.GlobalGP, self.vs, trace=trace, jit=jit, iters=iters)
        
    def predict(self, X_new): # Predict at new locations
        l_list = self.LocalGP(self.vs, self.X)
        global_ls = tf.concat(l_list, axis=1)
        
        l_list_new = self.LocalGP(self.vs, X_new)
        global_ls_new = tf.concat(l_list_new, axis=1)
        
        K = self.vs['global_gp_std']**2 * NSEQ(global_ls, global_ls)(self.X, self.X)
        K_star = self.vs['global_gp_std']**2 * NSEQ(global_ls_new, global_ls)(X_new, self.X)
        K_star_star = self.vs['global_gp_std']**2 * NSEQ(global_ls_new, global_ls_new)(X_new, X_new)
        
        L = B.cholesky(K + B.eye(self.X.shape[0]) * self.vs['global_gp_noise_std']**2)
        alpha = B.cholesky_solve(L, self.y)
        
        pred_mean = K_star@alpha
        
        v = B.cholesky_solve(L, B.T(K_star))
        pred_var = K_star_star + B.eye(X_new.shape[0])*self.vs['global_gp_noise_std']**2 - K_star@v
        
        return pred_mean, pred_var