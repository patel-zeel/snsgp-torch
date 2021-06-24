import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from model import NSGPRegression
import warnings
warnings.filterwarnings('ignore')

n = 11
np.random.seed(0)
x1 = np.sort(np.random.uniform(-0.5, 1, n))
x2 = np.sort(np.random.uniform(-0.5, 1, n))
X1, X2 = np.meshgrid(x1, x2)

def simulate(a, b):
    bi = np.pi * (2*a + 0.5*b + 1)
    return 0.1 * (np.sin(a*bi) + np.sin(b*bi))

y = np.array([simulate(a,b) for a,b in zip(X1.ravel(), X2.ravel())]).reshape(-1,1) + np.random.normal(0,0.025, n*n).reshape(-1,1)
X = np.array([(a,b) for a,b in zip(X1.ravel(), X2.ravel())])

print(X.shape, y.shape)

np.random.seed(0)
n = 31
x1_test = np.linspace(-0.5, 1, n)
X1_test, X2_test = np.meshgrid(x1_test, x1_test)

X_test = np.array([(a,b) for a,b in zip(X1_test.ravel(), X2_test.ravel())])
y_test = np.array([simulate(a,b) for a,b in zip(X1_test.ravel(), X2_test.ravel())]).reshape(-1,1)
print(X_test.shape, y_test.shape)

###################
Xscaler = StandardScaler()
yscaler = StandardScaler()

X_scaled = Xscaler.fit_transform(X)
y_scaled = yscaler.fit_transform(y)

X_test_scaled = Xscaler.transform(X_test)

###################
num_inducing_points = 3
seed = 0
f_ind = lambda X, num_ind: KMeans(n_clusters=num_ind, random_state=seed).fit(X_scaled, y_scaled).cluster_centers_
model = NSGPRegression(X_scaled, y_scaled, num_inducing_points, f_indu=f_ind, seed=0)
model.optimize(trace=True)

mean, var = model.predict(X_test_scaled)
pred_y = yscaler.inverse_transform(mean.mat)
print(np.mean(np.square(y_test-pred_y))/np.var(y_test))

plt.contourf(X1_test, X2_test, y_test.reshape(*X1_test.shape), levels=30)
plt.colorbar()
plt.savefig('test.jpg')
plt.figure()
plt.contourf(X1_test, X2_test, pred_y.reshape(*X1_test.shape), levels=30)
plt.colorbar()
plt.savefig('pred.jpg')
plt.figure()
plt.contourf(X1_test, X2_test, pred_y.reshape(*X1_test.shape)-y_test.reshape(*X1_test.shape), levels=30)
plt.colorbar()
plt.savefig('error.jpg')