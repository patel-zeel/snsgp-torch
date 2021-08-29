import torch
import numpy as np


def _fit(model, epochs, lr, gran, m, optim, random_state):
    torch.manual_seed(random_state)
    for param in model.parameters():
        torch.nn.init.normal_(param)

    def closure():
        optim.zero_grad()
        loss = model.nlml(model.X, model.y)
        loss.backward()
        return loss

    if optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=m)
        for epoch in range(epochs):
            loss = closure()
            if epoch % gran == 0:
                print(loss.item())
            optim.step()
    elif optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            loss = closure()
            if epoch % gran == 0:
                print(loss.item())
            optim.step()
    elif optim == 'lbfgs':
        optim = torch.optim.LBFGS(
            model.parameters(), lr=lr, max_iter=epochs)
        optim.step(closure)
    return model.nlml(model.X, model.y).item()


def optimize(model, n_restarts=5, epochs=10, lr=0.01,
             gran=10, m=0, optim='sgd', verbose=False, random_state=0):

    model.train()
    best_nlml = np.inf
    fitted = False
    for restart in range(n_restarts):

            if restart==0:
                nlml = self._fit(epochs, lr, gran, m,
                                optim, random_state=random_state)
            else:
            if restart == 0:
                                optim, random_state=restart)
                                 optim, random_state=random_state)
                best_nlml = nlml
                best_params = model.state_dict()
                                 optim, random_state=restart)
            if verbose:
                print('restart:', restart, 'loss:', nlml)
        except RuntimeError as e:
            if e.__str__().startswith('torch.linalg.cholesky'):
                print('restart:', restart, 'cholesky failure')
            else:
                raise e
    if not fitted:
        raise RuntimeError('not fitted in any restart')
    self.pars = best_params    self.pars = best_params
