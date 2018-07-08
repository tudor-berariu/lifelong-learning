import torch
from torch import Tensor
from typing import Iterator, List, Tuple

# Project imports
from agents.base_agent import BaseAgent


class SparsityPrior:
    def __init__(self, p: float = 1.0) -> None:
        self.p = p
        pass

    def __call__(self, vector: Iterator[Tuple[str, Tensor]]) -> Tensor:
        all_p = torch.cat([values.view(-1) for (_, values) in vector], dim=0)
        return torch.norm(all_p, p=self.p)


class Sparsity(BaseAgent):
    def __init__(self, *args, **kwargs):
        super(Sparsity, self).__init__(*args, **kwargs)

        args = self._args
        agent_args = args.lifelong

        self.sparsity_scale: float = agent_args.sparsity_scale
        self.sparsity_task_decay: float = agent_args.sparsity_task_decay
        self.sparsity_p_norm: float = agent_args.sparsity_p_norm

        if self.sparsity_scale > 0:
            self.sparsity_prior = SparsityPrior(p=self.sparsity_p_norm)

    def _train_task_batch_extra_losses(self, *args, **kwargs)-> List[torch.Tensor]:
        extra_losses = super(Sparsity, self)._train_task_batch_extra_losses(*args, **kwargs)

        sparsity_loss = 0
        if self.sparsity_scale > 0:
            sparsity_loss += self.sparsity_prior(self._model.named_parameters())
            sparsity_loss *= self.sparsity_scale
            extra_losses += [sparsity_loss]

        return extra_losses

    def _end_train_task(self):
        super(Sparsity, self)._end_train_task()
        self.sparsity_scale *= self.sparsity_task_decay
