from sklearn.cluster import KMeans
import torch
import numpy as np


class InducingFunctions(object):
    def __init__(self, device='cpu'):
        self.device = device

    def f_kmeans(self, X, num_inducing_points, random_state=None):
        model = KMeans(n_clusters=num_inducing_points,
                       random_state=random_state)
        out = model.fit(X).cluster_centers_
        return torch.tensor(out, dtype=X.dtype, device=self.device)

    def f_random(self, X, num_inducing_points, random_state=None):
        np.random.seed(random_state)
        inds = np.random.choice(
            X.shape[0], replace=False, size=num_inducing_points)
        return torch.tensor(X[inds], dtype=X.dtype, device=self.device)
