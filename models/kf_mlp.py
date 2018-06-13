from typing import List, Union
from functools import reduce
from operator import mul

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn import Module

from models.kf import KroneckerFactored


class KFMLP(KroneckerFactored):

    def __init__(self, cfg,
                 in_size: torch.Size,
                 out_sizes: List[int]) -> None:

        super(KFMLP, self).__init__()

        self.__use_softmax: bool = cfg.use_softmax
        hidden_units: List[int] = cfg.hidden_units
        activation = getattr(nn, cfg.activation)
        in_units = reduce(mul, in_size, 1)

        self.my_modules = []  # type: List[Module]

        for idx, hidden_size in enumerate(hidden_units):
            linear_layer = nn.Linear(in_units, hidden_size)
            setattr(self, f"linear_{idx:d}", linear_layer)
            transfer_layer = activation()
            setattr(self, f"transfer_{idx:d}", transfer_layer)
            self.my_modules.extend([linear_layer, transfer_layer])
            in_units = hidden_size

        self.heads = []  # type List[Module]
        for out_size in out_sizes:
            linear_layer = nn.Linear(in_units, out_size)
            setattr(self, f"head_{idx:d}", linear_layer)
            self.heads.append(linear_layer)

    def forward(self, *xs: List[torch.Tensor],
                head_idx: Union[int, Tensor] = 0) -> List[Tensor]:
        x: torch.Tensor = xs[0]

        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        for module in self.my_modules:
            x = module(x)

        assert not self.do_kf or isinstance(head_idx, int)  # kf -> single head

        results = []  # type: List[Tensor]
        if isinstance(head_idx, int):
            results.append(self.heads[head_idx](x))
        elif isinstance(head_idx, Tensor):
            assert head_idx.size(0) == x.size(0)  # TODO: remove this
            for task_idx in set(head_idx.tolist()):
                results.append(self.heads[task_idx](x[head_idx == task_idx]))
        else:
            raise TypeError

        if self.use_softmax:
            return [functional.softmax(y, dim=-1) for y in results]

        return results

    @property
    def use_softmax(self):
        return self.__use_softmax

    @use_softmax.setter
    def use_softmax(self, use_softmax):
        self.__use_softmax = use_softmax
