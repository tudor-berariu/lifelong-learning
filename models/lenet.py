import torch
import torch.nn as nn
from typing import List
import numpy as np
import torch.nn.functional as functional


class LeNet(nn.Module):

    def __init__(self, cfg,
                 in_size: torch.Size,
                 out_sizes: List[int]) -> None:
        super(LeNet, self).__init__()

        self.__use_softmax: bool = cfg.use_softmax
        hidden_units: List[int] = cfg.hidden_units

        self.features = nn.Sequential(
            nn.Conv2d(3, 20, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, 5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        in_units = self.get_flat_fts(in_size, self.features)
        hidden_layers = []
        for hidden_size in hidden_units:
            hidden_layers.append(nn.Linear(in_units, hidden_size))
            hidden_layers.append(nn.ReLU())
            in_units = hidden_size
        self.fc = nn.Sequential(*hidden_layers)

        self.heads = nn.ModuleList()
        for out_size in out_sizes:
            self.heads.append(nn.Linear(in_units, out_size))

    @staticmethod
    def get_flat_fts(in_size: torch.Size, fts: nn.Module) -> int:
        f = fts(torch.ones(1,*in_size))
        return int(np.prod(f.size()[1:]))

    def forward(self, *xs: List[torch.Tensor], head_idx: int = 0) -> torch.Tensor:
        x: torch.Tensor = xs[0]
        batch_size = x.size(0)

        x = self.features(x)
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
