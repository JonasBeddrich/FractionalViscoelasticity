
import torch

"""
==================================================================================================================
Mean Square Error objective function
==================================================================================================================
"""

class MSE:

    def __init__(self, data=0, start=0):
        self.data = data
        self.start = start
        if not torch.is_tensor(self.data):
            self.data = torch.tensor(self.data)
        if len(self.data.shape)==1:
            self.data = self.data.reshape([-1,1])
            self.two_loads = False
        else:
            self.two_loads = True

    def __call__(self, y):
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        x = y[self.start:]
        # weights = torch.tesnor([1, 10]).double()
        # J = torch.sum( (y - self.data).square().sum(dim=0) / self.data.square().sum(dim=0) )
        J = (x - self.data).square().sum(dim=0) / self.data.square().sum(dim=0)
        if self.two_loads:
            J = J[...,0] + J[...,1]
        return 0.5*J
