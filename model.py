from typing import List
from functools import reduce
from operator import mul
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.autograd import Variable

from my_types import Args, Model

from liftoff.config import value_of, namespace_to_dict


class MLP(nn.Module):

    def __init__(self, in_size: torch.Size,
                 hidden_units: int,
                 use_softmax: bool = True) -> None:

        super(MLP, self).__init__()
        in_units = reduce(mul, in_size, 1)
        self.fc1 = nn.Linear(in_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 10)
        self.use_softmax = use_softmax

    def forward(self, *xs: List[Variable]) -> Variable:
        inputs: Variable = xs[0]

        batch_size = inputs.size(0)
        inputs = inputs.view(batch_size, -1)
        output = self.fc2(F.relu(self.fc1(inputs)))
        if self.use_softmax:
            return F.softmax(output, dim=-1)
        return output

    @property
    def use_softmax(self):
        return self.__use_softmax

    @use_softmax.setter
    def use_softmax(self, use_softmax):
        self.__use_softmax = use_softmax


def get_model(args: Args) -> Model:
    return MLP(args.in_size, args.model.hidden_units[0])


def get_optimizer(model: Model, args: Args) -> Optimizer:
    kwargs = value_of(args, "optimizer_args", Namespace(lr=0.001))
    kwargs = namespace_to_dict(kwargs)
    return optim.__dict__[args.optimizer](model.parameters(), **kwargs)
