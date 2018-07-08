from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as functional

from torch import Tensor
from torch.nn import Module

from models.kfac import KFACModule


class KFLeNet(KFACModule):

    def __init__(self, cfg,
                 in_size: torch.Size,
                 out_sizes: List[int],
                 **kwargs) -> None:
        super(KFLeNet, self).__init__(**kwargs)

        self.__use_softmax: bool = cfg.use_softmax
        hidden_units: List[int] = cfg.hidden_units
        conv_config = cfg._conv
        activation = getattr(nn, cfg.activation)

        self.conv_modules = []  # type: List[Module]
        in_conv = in_size[0]
        ch, h, w = in_size

        for idx, layer_config in enumerate(conv_config):
            if len(layer_config) > 1:
                if layer_config[1] == -1:  # TODO: change this
                    layer_config[1] = in_conv
            conv_layer = getattr(nn, layer_config[0])(*layer_config[1:])

            if isinstance(conv_layer, nn.Conv2d):
                setattr(self, f"conv_{idx:d}", conv_layer)
                s_h, s_w = conv_layer.stride
                k_h, k_w = conv_layer.kernel_size
                p_h, p_w = conv_layer.padding
                h, w = (h - k_h + p_h) // s_h + 1, (w - k_w + p_w) // s_w + 1  # TODO: check paddinng
                ch = layer_config[2]
            else:
                if isinstance(conv_layer, nn.MaxPool2d):
                    s = conv_layer.stride
                    k = conv_layer.kernel_size
                    h, w = (h - k) // s + 1, (w - k) // s + 1  # TODO: check
                setattr(self, f"other_conv_{idx:d}", conv_layer)
            self.conv_modules.append(conv_layer)

        self.fc_modules = []  # type: List[Module]

        in_units = ch * h * w

        for idx, hidden_size in enumerate(hidden_units):
            linear_layer = nn.Linear(in_units, hidden_size)
            setattr(self, f"linear_{idx:d}", linear_layer)
            transfer_layer = activation()
            setattr(self, f"transfer_{idx:d}", transfer_layer)
            self.fc_modules.extend([linear_layer, transfer_layer])
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

        for module in self.conv_modules:
            x = module(x)
        x = x.view(batch_size, -1)
        for module in self.fc_modules:
            x = module(x)

        assert not self.kf_mode or isinstance(head_idx, int)  # kf -> single head

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
