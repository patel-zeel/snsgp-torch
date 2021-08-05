import pytest
import matplotlib.pyplot as plt
from time import time


def test_model():
    from nsgp import NSGP
    from nsgp.utils import InducingFunctions
    import torch
    import numpy as np
    device = 'cuda'

    num_low = 25
    num_high = 25
    gap = -.1
    noise = 0.0001
    X = torch.vstack((torch.linspace(-1, -gap/2.0, num_low)[:, np.newaxis],
                      torch.linspace(gap/2.0, 1, num_high)[:, np.newaxis])).reshape(-1, 1)
    y = torch.vstack((torch.zeros((num_low, 1)), torch.ones((num_high, 1))))
    scale = torch.sqrt(y.var())
    offset = y.mean()
    y = ((y-offset)/scale).reshape(-1, 1)

    X_new = torch.linspace(-1, 1, 100).reshape(-1, 1)

    X = X.to(device)
    y = y.to(device)
    X_new = X_new.to(device)

    indu = InducingFunctions(device)
    model = NSGP(X, y, f_inducing=indu.f_kmeans,
                 num_inducing_points=5, jitter=10**-5)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    # optim = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = []

    # Move it to device
    model.to(device)

    init = time()
    model.train()
    for _ in range(200):
        optim.zero_grad()
        loss = model.nlml()
        losses.append(loss.item())
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        y_new, y_var = model.predict(X_new)
        y_std2 = y_var.diagonal()**0.5 * 2
        print(time()-init, 'seconds')
        # fig, ax = plt.subplots(3, 1, figsize=(10, 16))
        # ax[0].scatter(X, y)
        # ax[0].plot(X_new.numpy(), y_new.numpy())
        # ax[0].fill_between(X_new.ravel(), y_new.ravel() -
        #                    y_std2, y_new.ravel()+y_std2, alpha=0.5)

        # ax[2].plot(losses)

        # ax[1].plot(X_new, model.get_LS(X_new)[0])

        # fig.savefig('./test_step_function.pdf')


if __name__ == '__main__':
    test_model()
