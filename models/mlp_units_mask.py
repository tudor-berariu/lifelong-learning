from typing import List, Union

import torch
from torch import Tensor
import torch.nn.functional as functional

from .mlp import MLP


class MaskedMLP(MLP):

    def __init__(self, *args, **kwargs) -> None:
        super(MaskedMLP, self).__init__(*args, **kwargs)
        self.crt_mask_idx = None
        self.crt_mask = None
        self.activation_mask = dict()

    def set_mask(self, mask_idx: int, mask: torch.Tensor, reset: bool =False):
        self.crt_mask_idx = mask_idx
        self.crt_mask = mask
        if mask_idx not in self.activation_mask or reset:
            self.activation_mask.pop(mask_idx, None)
            self.activation_mask[mask_idx] = act_mask = list()

            for hidden_size in self.hidden_units:
                layer_mask = torch.ones(hidden_size).to(device=mask.device)
                layer_mask[mask] = 0
                act_mask.append(layer_mask)

    def forward(self, *xs: List[torch.Tensor],
                head_idx: Union[int, Tensor] = 0) -> List[Tensor]:

        apply_mask = self.crt_mask_idx is not None
        masks = None
        if apply_mask:
            masks = self.activation_mask[self.crt_mask_idx]

        x: torch.Tensor = xs[0]

        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        linear_idx = 0
        for ix, layer in enumerate(self.fc):
            x = layer(x)
            if apply_mask and isinstance(layer, torch.nn.Linear):
                x = x * masks[linear_idx]
                linear_idx += 1

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
