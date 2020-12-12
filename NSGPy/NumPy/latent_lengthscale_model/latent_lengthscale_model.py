## Terminology is consistent with paper 
# Paper title: Nonstationary Gaussian Process Regression Using Point Estimates of Local Smoothness
# Authors: Christian Plagemann, Kristian Kersting, and Wolfram Burgard

#import jax.numpy as np
import numpy as np
from scipy.optimize import minimize
# from jax import jit, grad
# from jax.scipy.optimize import minimize
from sklearn.cluster import KMeans
import scipy

class LLS:
    """
    Latent lengthscale kernel
    We have the "main GP" and the "lengthscale GP" to model the lengthscales of the "main GP"
    
    input_dim: int, Input dimentions 
    
    sigma_f: float, Main GP variance 
    
    sigma_n: float, Main GP noise varaince
    
    N_l_bar: int, Number of support points (x_bar) where lengthscale will be learnt
    
    sigma_f_bar: float, Lengthscale GP variance
    
    sigma_l_bar: float, Lengthscale GP lengthscale
    
    sigma_n_bar: float, Lengthscale GP noise variance
    
    l_bar: numpy.ndarray, Lenthscales vector at support points (x_bar)
    
    optimizer: str, ('scipy', 'jax'), optimizer to use for hyperparameter learning 
    
    n_iter: int, Number of iterations (applicable on some optimizers)
    
    lr: float, learning rate (applicable on some optimizers)
    
    kernel: str, kernal name to use, 'rbf' or 'matern'
    
    nu: float, a hyperparameter in 'matern' kernel
    
    pruned_kernel: Under development
    
    full_l_mat: Under development
    
    N_l_bar_method: Under development
    
    """
    def __init__(self, input_dim, sigma_f=None, sigma_n=None, N_l_bar=5,
                 sigma_f_bar=None, sigma_l_bar=None, sigma_n_bar=None, l_bar=None, 
                 optimizer='scipy', n_iter=20, lr=0.1, seed=0, kernel='rbf', nu=1.5, l_isotropic=False,
                pruned_kernel=False, full_l_mat=False, N_l_bar_method='uniform', store_history=True):
        
        self.seed = seed                              # random seed
        np.random.seed(self.seed)
        self.input_dim = int(input_dim)               # Input dimentions 
        self.sigma_f = np.abs(np.random.rand(1)) if sigma_f is None else np.array(sigma_f).reshape(1,)  
                                                      # Main GP variance 
        self.sigma_n = np.abs(np.random.rand(1)) if sigma_n is None else np.array(sigma_f).reshape(1,)
                                                      # Main GP noise varaince
        self.N_l_bar = int(N_l_bar)                   # Number of support points (x_bar) where lengthscale will be learnt
        self.l_isotropic = l_isotropic                # Lengthscale GP is isotropic or not
        if self.l_isotropic:
            self.sigma_f_bar = np.abs(np.random.rand(1)) if sigma_f_bar is None else np.array(sigma_f_bar) 
                                                   # Lengthscale GP variance
            self.sigma_l_bar = np.abs(np.random.rand(1)) if sigma_l_bar is None else np.array(sigma_l_bar)
                                                   # Lengthscale GP lengthscale
            self.sigma_n_bar = np.abs(np.random.rand(1)) if sigma_n_bar is None else np.array(sigma_n_bar)
                                                   # Lengthscale GP noise variance
            assert self.sigma_f_bar.size == 1
            assert self.sigma_l_bar.size == 1
            assert self.sigma_n_bar.size == 1
            
        else:
            self.sigma_f_bar = np.abs(np.random.rand(1)) if sigma_f_bar is None else np.array(sigma_f_bar)
            self.sigma_l_bar = np.abs(np.random.rand(input_dim)) if sigma_l_bar is None else np.array(sigma_l_bar)
            self.sigma_n_bar = np.abs(np.random.rand(1)) if sigma_n_bar is None else np.array(sigma_n_bar)
            assert self.sigma_f_bar.size == 1
            assert self.sigma_l_bar.size == input_dim
            assert self.sigma_n_bar.size == 1
        
        self.l_bar = np.abs(np.random.rand(N_l_bar, input_dim)) if l_bar is None else np.array(l_bar)
                                                   # lenthscales at support points (x_bar)
        assert type(self.sigma_f_bar) == type(np.ones(1))
        assert type(self.sigma_l_bar) == type(np.ones(1))
        assert type(self.sigma_n_bar) == type(np.ones(1))
        assert self.l_bar.shape == (N_l_bar, input_dim)
        
        self.optimizer = optimizer           # Optimizer to use for parameters learning
        self.n_iter = n_iter                 # Number of iterations
        self.lr = lr                         # Learning rate
        self.store_history = store_history   # Store history (bool)
        
        self.kernel = kernel                 # Kernel to use for main GP (Not implemented yet)
        self.nu = nu                         # Exponent for Matern kernel (Not implemented yet)
        self.pruned_kernel = pruned_kernel   # Use unnormalized Main GP kernel (Not implemented yet)
        self.full_l_mat = full_l_mat         # Use complete lengthscale matrix (Not implemented yet)
        self.N_l_bar_method = N_l_bar_method # Method to choose support points (Not implemented yet)
    
    def K_bar(self, sigma_f_bar, sigma_l_bar, X_bar_1, X_bar_2=None): # RBF kernel for lengthscale GP for training
        """
        X_bar_1, X_bar_2: np.ndarray with shape (N_l_bar, input_dim)
        sigma_f, sigma_n: int, variance (unsquared) and noise variance (unsquared)
        """
        if X_bar_2 is None:
            X_bar_2 = X_bar_1
            
        d = np.square(X_bar_1[:, np.newaxis, :] - X_bar_2[np.newaxis, :, :]) # 3D vectorized distance
        
        if self.l_isotropic: # If lengthscale GP is isotropic
            k = np.exp(-d/(sigma_l_bar**2))
        else:
            k = np.exp(-d/(sigma_l_bar.reshape(1,1,self.input_dim)**2))
        
        return k*sigma_f_bar**2
        
    def K_bar_(self, X_bar_1, X_bar_2=None): # RBF kernel for lengthscale GP for prediction
        return self.K_bar(self.sigma_f_bar, self.sigma_l_bar, X_bar_1, X_bar_2)
    
    def predict_lengthscales(self, X, sigma_f_bar, sigma_l_bar, 
                             sigma_n_bar, l_bar, return_L_bar=False): # This method is used for training
        
        self.K__X_bar__X_bar = self.K_bar(sigma_f_bar, sigma_l_bar, self.X_bar, X_bar_2=None)
        self.K__X__X_bar = self.K_bar(sigma_f_bar, sigma_l_bar, X, self.X_bar)
        
        alphaL = []
        L_barL = []
        p,q,r = self.K__X_bar__X_bar.shape
        for dim in range(self.input_dim): # Oops, No vectorization available
            if self.l_isotropic:
#                 self.K__X_bar__X_bar = self.K__X_bar__X_bar.ravel().at[::p*r+r]\
#                 .add(sigma_n_bar).reshape(p,q,r)  # Add noise variance to diagonal
                self.K__X_bar__X_bar.ravel()[::p*r+r] += 10**-10#sigma_n_bar
                pass
            else:
#                 self.K__X_bar__X_bar = self.K__X_bar__X_bar.ravel().at[::p*r+r]\
#                 .add(sigma_n_bar[dim]).reshape(p,q,r)
                self.K__X_bar__X_bar.ravel()[::p*r+r] += 10**-10#sigma_n_bar#[dim]
                pass
            self.L_bar = np.linalg.cholesky(self.K__X_bar__X_bar[:,:,dim])           # Cholesky decomposition
            L_barL.append(self.L_bar[:,:,np.newaxis])
            alphaL.append(scipy.linalg.cho_solve((self.L_bar, True), np.log(l_bar[:,dim:dim+1]))[:,np.newaxis,:])
        self.alpha_bar = np.concatenate(alphaL, axis=1)
        self.L_bar = np.concatenate(L_barL, axis=2)
    
        l = np.exp(np.einsum('pqr,qrs->pr', self.K__X__X_bar, self.alpha_bar))
        
        if return_L_bar:
            return l, self.L_bar
        return l
    
    def predict_lengthscales_(self, X): # Predict lengthscales at predict time
        return self.predict_lengthscales(X, self.sigma_f_bar, self.sigma_l_bar, 
                                    self.sigma_n_bar, self.l_bar)
    
    def K(self, sigma_f, Xi, li, Xj=None, lj=None, diag=False): # From Eq. 7, Main kernel for training
        """
        for scaler inputs 
        K(x_i, x_j) = ((l_i^2)**0.25)*((l_j^2)**0.25)*(0.5**-0.5)*((l_i^2+l_j^2)**-0.5)*(exp(2*D/(l_i^2+l_j^2)))
        for vector inputs (NumPy syntax)
        K(X_i, X_j) = ((L_i^2@L_j.T^2)**0.25)*(0.5**-0.5)*((L_i^2+L_j.T^2)**-0.5)*(exp(2*D/(L_i^2+L_j.T^2)))
        for multidimentional case, everything multiply and log exponents sums up
        
        Xi, Xj: np.ndarray with shape (N, input_dim)
        li, lj: np.ndarray with shape (N, input_dim)
        sigma_f, sigma_n: int, variance (unsquared) and noise variance (unsquared)
        """
        
        l_sqr_i = np.square(li)
        if Xj is None:
            Xj = Xi
            l_sqr_j = l_sqr_i
        else:
            l_sqr_j = np.square(lj)

        if diag:
            P = np.square(l_sqr_i.prod(axis=1))    # 3d vectorized diagonal (l_i^2) * (l_i^2)
            P_s = 2 * l_sqr_i.prod(axis=1)         # 3d vectorized diagonal (l_i^2) + (l_i^2)
            k = np.sqrt(np.sqrt(P))/np.sqrt(P_s)
        else:
            P = l_sqr_i.prod(axis=1)[:,np.newaxis]@l_sqr_j.prod(axis=1)[np.newaxis,:]    # 3d vectorized (l_i^2) * (l_j^2)
            P_s = (l_sqr_i[:, np.newaxis, :] + l_sqr_j[np.newaxis, :, :]) # 3d vectorized (l_i^2) + (l_j^2)
            D = (np.square(Xi[:, np.newaxis, :] - Xj[np.newaxis, :, :])/P_s).sum(axis=2) # 3d vectorized d^2 / P_s
            k = ((np.sqrt(np.sqrt(P)))/(np.sqrt(P_s.prod(axis=2))))*np.exp(-2*D)  # Eq. 7, Put everything togather
            
        return k*(sigma_f**2*0.5**-0.5) # Multiplying scalers at the end
    
    def K_(self, Xi, li, Xj=None, lj=None, diag=False): # Main kernel for predictions
        return self.K(self.sigma_f, Xi, li, Xj, lj, diag)
        
    def parse_params(self, params):
        sigma_f = params[0:1]
        sigma_n = params[1:2]
        sigma_f_bar = params[2:3]
        if self.l_isotropic:
            sigma_l_bar = params[3:4]
            sigma_n_bar = params[4:5]
            l_bar = params[5:].reshape(self.X_bar.shape[0], self.input_dim)
        else:
            sigma_l_bar = params[3:3+self.input_dim]
            sigma_n_bar = params[3+self.input_dim:3+self.input_dim+1]
            l_bar = params[3+self.input_dim+1:].reshape(self.X_bar.shape[0], self.input_dim)
        
        return sigma_f, sigma_n, sigma_f_bar, sigma_l_bar, sigma_n_bar, l_bar
        
    def mll(self, params): # Marginal log likelihood from equation 6 in section 4.1 
        # parsing parameters
        sigma_f, sigma_n, sigma_f_bar, sigma_l_bar, sigma_n_bar, l_bar = self.parse_params(params)
        
        # Calculations for lengthscale GP
        self.l, self.L_bar = self.predict_lengthscales(self.X, sigma_f_bar, sigma_l_bar,
                                      sigma_n_bar, l_bar, return_L_bar=True)
        
        # Calculations for Main GP
        self.K_XX = self.K(sigma_f, self.X, self.l, Xj=None, lj=None)
        p,q = self.K_XX.shape

        self.K_XX.ravel()[::p+1] += sigma_n
        
        self.L = np.linalg.cholesky(self.K_XX)
        self.alpha = scipy.linalg.cho_solve((self.L, True), self.y)
        
        # alpha -> (self.X.shape[0], 1)
        # alpha_bar -> (self.X_bar.shape[0], input_dim)
        L_theta =  (self.y.T@self.alpha).squeeze() +\
                    np.sum(np.log(np.diag(self.L)))
        for dim in range(self.input_dim):
            L_theta += np.sum(np.log(np.diag(self.L_bar[:,:,dim])))
        #print(L_theta.shape)
        return L_theta
    
    def get_mll(self, sigma_f=None, sigma_n=None, sigma_f_bar=None, sigma_l_bar=None, sigma_n_bar=None, l_bar=None):
        params = [sigma_f, sigma_n, sigma_f_bar, sigma_l_bar, sigma_n_bar, l_bar]
        fitted_params = [self.sigma_f, self.sigma_n, self.sigma_f_bar, self.sigma_l_bar, self.sigma_n_bar, self.l_bar.ravel()]
        for p_i in range(len(params)):
            if params[p_i] is None:
                params[p_i] = fitted_params[p_i]
        return self.mll(np.concatenate(params))

    def fit(self, X, y):
        assert len(X.shape) == 2, "shape of X must be 2D"
        assert len(y.shape) == 2, "shape of y must be 2D"
        assert y.shape[1] == 1, "y must be of shape (*,1)"

        self.X = X
        self.y = y
        
        if self.N_l_bar <= self.X.shape[0]:
            kmeans = KMeans(n_clusters=self.N_l_bar, random_state=0)
            self.X_bar = np.array(kmeans.fit(X).cluster_centers_).reshape(self.N_l_bar, self.input_dim)
        else:
            self.X_bar = X
        
        params = np.concatenate([self.sigma_f, self.sigma_n, self.sigma_f_bar,
                                     self.sigma_l_bar, self.sigma_n_bar, self.l_bar.ravel()])
#         print('initials',params)

        if self.optimizer=='scipy':
            res = scipy.optimize.minimize(self.mll, params.tolist(), 
                                          bounds=[(10**-5,10**5) for _ in range(len(params))])
            params = res.x
            self.sigma_f, self.sigma_n, self.sigma_f_bar,\
            self.sigma_l_bar, self.sigma_n_bar, self.l_bar = self.parse_params(params)
        
        elif self.optimizer=='jax': # add jit later
            self.history = {'loss':[], 'grads':[]}
            for iteration in range(self.n_iter):
                grads = grad(self.mll)(params)
                params = np.array(params - self.lr*grads).clip(10**-5, 10**5)
                if self.store_history:
                    self.history['loss'].append(self.mll(params)) # remove after verification
                    self.history['grads'].append(grads)
            self.sigma_f, self.sigma_n, self.sigma_f_bar,\
            self.sigma_l_bar, self.sigma_n_bar, self.l_bar = self.parse_params(params)
        
        optim_fun_val = self.mll(params)
    
        self.params = {'likelihood (mll)':optim_fun_val, 'GP_variance (sigma_f)':self.sigma_f, 
                   'GP_noise_level (sigma_n)':self.sigma_n, 'L_GP_variance (sigma_f_bar)':self.sigma_f_bar, 
                   'L_GP_lengthscale (sigma_l_bar)':self.sigma_l_bar, 'L_GP_noise_level (sigma_n_bar)':self.sigma_n_bar,
                  'N_lengthscales (l_bar)':self.l_bar}

    def get_params(self):
        return self.params

    def predict(self, X_star, return_cov=True, diag=True):
        l_star = self.predict_lengthscales_(X_star)
        K__X_star__X = self.K_(X_star, l_star, self.X, self.l)
        
        mean = K__X_star__X@self.alpha
        if return_cov:
            v = scipy.linalg.cho_solve((self.L, True), K__X_star__X.T)
            if diag:
                cov = self.K_(X_star, l_star, diag=True) - (K__X_star__X@v).diagonal()
            else:
                cov = self.K_(X_star, l_star) - K__X_star__X@v
            return mean, cov
        return mean
