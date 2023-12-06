import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

import models
from models import register


@register('coordinate-embedding')
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.out_dim = channel
        self.input = nn.Parameter(torch.randn(1, size**2, channel))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1)
        return out

        
class ConLinear(nn.Module):
    def __init__(self, ch_in, ch_out, is_first=False, bias=True):
        super(ConLinear, self).__init__()
        self.linear = nn.Linear(ch_in, ch_out, bias=bias)
        if is_first:
            nn.init.uniform_(self.linear.weight, -np.sqrt(9 / ch_in), np.sqrt(9 / ch_in))
        else:
            nn.init.uniform_(self.linear.weight, -np.sqrt(3 / ch_in), np.sqrt(3 / ch_in))

    def forward(self, x):
        return self.linear(x)


class SinActivation(nn.Module):
    def __init__(self,):
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


@register('fourier-features')
class LFF(nn.Module):
    def __init__(self, hidden_size, ):
        super(LFF, self).__init__()
        self.out_dim = hidden_size
        self.ffm = ConLinear(2, hidden_size, is_first=True)
        self.activation = SinActivation()

    def forward(self, x):
        x = self.ffm(x)
        x = self.activation(x)
        return x