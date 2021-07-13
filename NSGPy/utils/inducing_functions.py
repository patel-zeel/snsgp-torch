import lab as B
from sklearn.cluster import KMeans
import numpy as np


def f_kmeans(X, num_inducing_points, random_state=None):
    return KMeans(n_clusters=num_inducing_points, random_state=random_state).fit(X).cluster_centers_


def f_random(X, num_inducing_points, random_state=None):
    np.random.seed(random_state)
    inds = np.random.choice(
        X.shape[0], replace=False, size=num_inducing_points)
    return X[inds]
