import torch
import numpy as np
from scipy.special import binom
from torch.nn import Module



class Bezier(Module):
    """
    Formulation of the Bezier curve
    """
    def __init__(self, num_bends):
        super(Bezier, self).__init__()
        self.register_buffer(
            "binom",
            torch.Tensor(binom(num_bends - 1, np.arange(num_bends), 
            dtype=np.float32))
        )
        self.register_buffer("range", torch.arange(0, float(num_bends)))
        self.register_buffer("rev_range", torch.arange(float(num_bends - 1), -1, -1))


    def forward(self, t):
        return self.binom * torch.pow(t, self.range) * torch.pow((1.0 - t), self.rev_range)



class PolyChain(Module):
    """
    Formulation of the poly-chain
    """
    def __init__(self, num_bends):
        super(PolyChain, self).__init__()
        self.num_bends = num_bends
        self.register_buffer("range", torch.arange(0, float(num_bends)))


    def forward(self, t):
        t_n = t * (self.num_bends - 1)
        return torch.max(self.range.new([0.0]), 1.0 - torch.abs(t_n - self.range))