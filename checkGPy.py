import numpy as np
import GPy

X = np.random.rand(10,2)
y = np.random.rand(10,1)

model = GPy.models.GPRegression(X, y, GPy.kern.RBF(X.shape[1], ARD=True))

model.optimize_restarts(10)

print(model.predict(X))