import torch


class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        A = torch.empty(1, 3, device='cpu')
        self.A = torch.nn.Parameter(A, requires_grad=False)
        pass

    def forward(self, x):
        return x * self.A


module = MyModule()
module.cuda()
for param in module.parameters():
    print(param)
