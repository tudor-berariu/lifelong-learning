from typing import Dict, Iterator, List, Tuple, Union

import torch
import torch.nn.functional as functional
from torch import Tensor

from termcolor import colored as clr

# Project imports
from models.kf import KFHessianProduct, KroneckerFactored
from .base_agent import BaseAgent


class Prior(object):

    def __call__(self, vector: Iterator[Tuple[str, Tensor]]) -> Tensor:
        raise NotImplementedError


class GaussianPrior(Prior):

    def __init__(self,
                 mode: Iterator[Tuple[str, Tensor]],
                 kfhp: KFHessianProduct,
                 diag_adjust: float = 0.) -> None:
        self.mode = {name: param.clone().detach_() for (name, param) in mode}
        self.kfhp = kfhp
        self.diag_adjust = diag_adjust

    def __call__(self, vector: Iterator[Tuple[str, Tensor]]) -> Tensor:
        all_diffs = dict({})
        diag_loss = None
        for name, values in vector:
            all_diffs[name] = diff = values - self.mode[name]
            if self.diag_adjust > 0:
                diag_extra = torch.dot(diff.view(-1), diff.view(-1))
                diag_loss = diag_extra if diag_extra is None else (diag_loss + diag_extra)
        vHv = self.kfhp.hessian_product_loss(all_diffs)
        if self.diag_adjust > 0:
            return vHv + self.diag_adjust * diag_loss
        return vHv


class SparsityPrior(Prior):

    def __init__(self, p: float = 1.0) -> None:
        self.p = p
        pass

    def __call__(self, vector: Iterator[Tuple[str, Tensor]]) -> Tensor:
        loss = None
        all_p = torch.cat([values.view(-1) for (_, values) in vector], dim=0)
        return torch.norm(all_p, p=self.p)


class SparseKFLaplace(BaseAgent):
    def __init__(self, *args, **kwargs):
        super(SparseKFLaplace, self).__init__(*args, **kwargs)
        assert isinstance(self._model, KroneckerFactored)

        args = self._args
        agent_args = args.lifelong

        self.merge_elasticities = agent_args.merge_elasticities  # TODO: not used yet
        self.nll_scale = agent_args.nll_scale
        self.sparsity_p_norm: float = agent_args.sparsity_p_norm
        self.prior_scale: float = agent_args.prior_scale
        self.sparsity_scale: float = agent_args.sparsity_scale
        self.sparsity_task_decay: float = agent_args.sparsity_task_decay
        self.diag_adjust: float = agent_args.diag_adjust  # (H + λI)
        self.use_exact: bool = agent_args.use_exact
        self.clamp_vector: bool = agent_args.clamp_vector

        self.saved_tasks_no = 0
        self.priors: Dict[str, Prior] = []

        if self.sparsity_scale > 0:
            self.sparsity_prior = SparsityPrior(p=agent_args.sparsity_p_norm)

    def _train_task_batch(self, batch_idx: int, data: Tensor, targets: List[Tensor],
                          head_idx: Union[int, Tensor])-> Tuple[List[Tensor], Tensor, Dict]:

        model = self._model
        self._optimizer.zero_grad()
        outputs = model(data, head_idx=head_idx)
        task_no = self.crt_task_idx

        losses = dict({})
        nll_loss = outputs[0].new_zeros(1)
        for out, target in zip(outputs, targets):
            nll_loss += functional.cross_entropy(out, target)

        sparsity_loss = nll_loss.new_zeros(1)
        if self.sparsity_scale > 0:
            sparsity_loss += self.sparsity_prior(model.named_parameters())

        prior_loss = nll_loss.new_zeros(1)
        if task_no > 0:
            for _idx, prior in enumerate(self.priors):
                prior_loss += prior(model.named_parameters())
        loss = self.prior_scale * prior_loss +\
            self.sparsity_scale * sparsity_loss +\
            self.nll_scale * nll_loss
        loss /= self.nll_scale   # This is not to search for new hyperparameter values

        losses = {
            "nll loss (raw)": nll_loss.item(),
            "sparsity p-norm (raw)": sparsity_loss.item(),
            "Gaussian prior (raw)": prior_loss.item(),
            "nll loss (scaled)": nll_loss.item() * self.nll_scale,
            "sparsity p-norm (scaled)": sparsity_loss.item() * self.sparsity_scale,
            "Gaussian prior (scaled)": prior_loss.item() * self.prior_scale,
            "FINAL LOSS:": loss.item()
        }
        loss.backward()
        self._optimizer.step()
        return outputs, loss, losses

    def _end_train_task(self):
        self.sparsity_scale *= self.sparsity_task_decay
        train_loader, _ = self.crt_data_loaders
        assert hasattr(train_loader, "__len__")
        self._optimizer.zero_grad()
        model = self._model
        model.use_exact = False
        model.do_kf = True
        for _batch_idx, (data, targets, head_idx) in enumerate(train_loader):
            model.zero_grad()
            outputs = model(data, head_idx=head_idx)
            loss = outputs[0].new_zeros(1)
            for output, target in zip(outputs, targets):
                loss += functional.cross_entropy(output, target)
            loss.backward()
        self.priors.append(GaussianPrior(model.named_parameters(), model.end_kf(),
                                         diag_adjust=self.diag_adjust))

        print(clr(f"There are {len(self.priors):d} Gaussian priors!", attrs=["bold"]))
