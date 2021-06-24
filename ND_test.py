# !pip uninstall numpy -y
# !pip install numpy
from varz import Vars
import GPy
import tensorflow as tf
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from sklearn.cluster import KMeans

# Importing models
from NSGPy.stheno.model import NSGPRegression as NSS
from NSGPy.torch.model import NSGPRegression as NST

### Common data and params

seed = 0
input_dim = 10
N = 100
num_inducing_points = 15

np.random.seed(seed)
rand = lambda shape: np.abs(np.random.normal(loc=0, scale=1, size=shape))
local_gp_std = rand((input_dim,))
local_gp_ls = rand((input_dim,))
local_ls = rand((input_dim, num_inducing_points))
local_gp_noise_std = rand((input_dim,))

global_gp_std = np.abs(np.random.normal())
global_gp_noise_std = np.abs(np.random.normal())

f_indu = lambda x, num_ind: KMeans(n_clusters=num_ind, random_state=seed).fit(x).cluster_centers_

X = np.random.rand(N,input_dim)
y = np.random.rand(N,1)

X_test = np.random.rand(N*2, input_dim)

### GPy

m = GPy.models.GPRegression(X, y, GPy.kern.RBF(input_dim, ARD=True))
GPy_pred_y, GPy_pred_var = m.predict(X_test, full_cov=True)

### Stheno model

vs = Vars(tf.float64)
# Local params
vs.positive(init=local_gp_std, shape=(input_dim,), name='local_gp_std')
vs.positive(init=local_gp_ls, shape=(input_dim,), name='local_gp_ls')
vs.positive(init=local_ls, 
                    shape=(input_dim, num_inducing_points), name='local_ls')
vs.positive(init=local_gp_noise_std, shape=(input_dim,), name='local_gp_noise_std')

# Global params
vs.positive(init=global_gp_std, name='global_gp_std')
vs.positive(init=global_gp_noise_std, name='global_gp_noise_std')

model = NSS(X, y, vs, num_inducing_points, f_indu, seed=seed)

stheno_pred_y, stheno_pred_var = model.predict(X_test)

# Torch
params = {}
params['local_gp_std'] = torch.tensor(local_gp_std)
params['local_gp_ls'] = torch.tensor(local_gp_ls)
params['local_gp_noise_std'] = torch.tensor(local_gp_noise_std)
params['local_ls'] = torch.tensor(local_ls)
params['global_gp_std'] = torch.tensor(global_gp_std)
params['global_gp_noise_std'] = torch.tensor(global_gp_noise_std)

model = NST(torch.tensor(X), torch.tensor(y), num_inducing_points, f_indu, params, seed)
torch_pred_y, torch_pred_var = model.predict(torch.tensor(X_test))

print(np.allclose(torch_pred_y, stheno_pred_y.mat))
print(np.allclose(torch_pred_var, stheno_pred_var.mat))

import matplotlib.pyplot as plt

plt.plot(torch_pred_y, 'd-')
plt.plot(stheno_pred_y.mat, 'o-')
plt.savefig('compare.jpg')

