from typing import List
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable


class MLP(nn.Module):

    def __init__(self, cfg,
                 in_size: torch.Size,
                 out_sizes: List[int]) -> None:

        super(MLP, self).__init__()

        self.__use_softmax: bool = cfg.use_softmax
        hidden_units: List[int] = cfg.hidden_units

        in_units = reduce(mul, in_size, 1)
        self.fc = nn.ModuleList()
        for hidden_size in hidden_units:
            self.fc.append(nn.Linear(in_units, hidden_size))
            self.fc.append(nn.ReLU())
            in_units = hidden_size

        self.heads = nn.ModuleList()
        for out_size in out_sizes:
            self.heads.append(nn.Linear(in_units, out_size))

    def forward(self, *xs: List[Variable], head_idx: int = 0) -> Variable:
        x: Variable = xs[0]

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        output = self.heads[head_idx](x)

        if self.use_softmax:
            return functional.softmax(output, dim=-1)

        return output

    @property
    def use_softmax(self):
        return self.__use_softmax

    @use_softmax.setter
    def use_softmax(self, use_softmax):
        self.__use_softmax = use_softmax
