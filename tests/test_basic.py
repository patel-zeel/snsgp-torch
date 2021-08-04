import pytest


def test_model():
    from nsgp import NSGP
    import torch

    X = torch.rand(1000, 3, dtype=torch.float32)*100
    y = (X[:, 0] + X[:, 1] + X[:, 2]).reshape(-1, 1) + torch.rand(1000, 1)

    X_new = torch.rand(10000, 3, dtype=torch.float32)

    model = NSGP(X, y, jitter=10**-5)
    optim = torch.optim.Adam(model.parameters(), lr=0.1)

    losses = []
    model.train()
    for _ in range(10):
        optim.zero_grad()
        loss = model.nlml()
        losses.append(loss.item())
        loss.backward()
        optim.step()

    print(losses)
    model.eval()
    y_new, y_var = model.predict(X_new)
    assert y_new.shape[0] == X_new.shape[0]
