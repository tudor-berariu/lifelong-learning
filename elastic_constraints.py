from typing import Iterable, List, Optional, Tuple
# from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable

from tasks import permute
from my_types import Args, Tasks

from liftoff.config import value_of
from torchutils import InMemoryDataLoader

Parameters = Iterable[Parameter]
Variables = Iterable[Variable]
Coefficients = Iterable[Iterable[Variable]]


class ElasticConstraint(object):

    def __init__(self, model: nn.Module,
                 tasks: Tasks,
                 learned: List[Tuple[str, int]],
                 args: Args):
        print("Creating an elastic constraint around current parameters.")
        # Copy reference paramters
        self.p_zero = [Variable(p.data.clone()) for p in model.parameters()]

        # Configure elastic loss
        self.is_signed = value_of(args.elasticity, "is_signed", False)
        self.loss_norm = value_of(args.elasticity, "loss_norm", 2)
        self.g_norm = value_of(args.elasticity, "g_norm", 2)
        self.drop_wrong = value_of(args.elasticity, "drop_misclassified", True)
        self.do_sample = value_of(args.elasticity, "do_sample", False)
        assert not self.do_sample or (0 < self.do_sample <= 1)

        # Coefficients will always pe positive
        self._coefficients = None
        self.used_no = {}
        if self.is_signed:
            self._compute_signed_coefficients(model, tasks, learned, args)
        else:
            self._compute_unsigned_coefficients(model, tasks, learned, args)

        # Normalize coeefficients
        norm = torch.cat([cff.data.view(-1) for cf_group in self.coefficients
                          for cff in cf_group])\
                    .norm(self.g_norm)
        for cf_group in self.coefficients:
            for cff in cf_group:
                cff.data.div_(max(norm, 1e-10))

    def _compute_unsigned_coefficients(self, model: nn.Module,
                                       tasks: Tasks,
                                       learned: List[Tuple[str, int]],
                                       args: Args) -> None:
        model.train()  # Model must be in train mode
        model.zero_grad()

        for (dataset, p_idx) in learned:
            task = tasks[dataset]
            i_perm = task.perms[0][p_idx]
            t_perm = None if task.perms[1] is None else task.perms[1][p_idx]
            used_no = 0
            loader: InMemoryDataLoader = task.train_loader
            old_bs, old_shuffle = loader.batch_size, loader.shuffle
            loader.batch_size, loader.shuffle = 0, False

            for data, target in loader:
                # Take just some samples, not all
                # This is not the smartest way to do this
                if self.do_sample:
                    chosen_mask = torch.rand(target.size()) < self.do_sample
                    chosen_idx = chosen_mask.nonzero()
                    if chosen_idx.nelement() > 0:
                        data = data.index_select(0, chosen_idx)
                        target = target.index_select(0, chosen_idx)
                    else:
                        continue

                # Perform forward step
                if args.cuda and not data.is_cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = permute(data, target, i_perm, t_perm)
                output = model(Variable(data))

                # Keep all or just the correctly classified ones
                if self.drop_wrong:
                    correct_mask = (output.data.max(1)[1] == target)
                    correct_idx = correct_mask.nonzero()
                    if correct_idx.nelement() > 0:
                        correct_idx.squeeze_(1)
                        output = output.index_select(0, Variable(correct_idx))
                        target = target.index_select(0, correct_idx)
                    else:
                        continue

                # Accumulate gradients
                loss, used_in_batch = self._loss(output, Variable(target))
                if used_in_batch > 0:
                    used_no += used_in_batch
                    loss *= .001
                    loss.backward()
                else:
                    print("WTF?")
            self.used_no[(dataset, p_idx)] = used_no

            loader.batch_size, loader.shuffle = old_bs, old_shuffle

            print(f"Used {used_no:d}/{len(task.train_loader.dataset):d} "
                  f"samples "
                  f"{(100.0 * used_no) / len(task.train_loader.dataset):f}% "
                  f"from {dataset:s}-{(p_idx+1):03d}.")

        self._coefficients = [[Variable(p.grad.data.abs())
                               for p in model.parameters()]]

    def _compute_signed_coefficients(self, model: nn.Module,
                                     tasks: Tasks,
                                     learned: List[Tuple[str, int]],
                                     args: Args) -> None:

        model.train()  # Model must be in training mode
        if args.elasticity.signed_on_cpu:
            model.cpu()  # Batches of one are faster on the CPU

            c_plus = [torch.zeros_like(p.data.cpu())
                      for p in model.parameters()]
            c_minus = [torch.zeros_like(p) for p in c_plus]
        else:
            c_plus = [torch.zeros_like(p.data) for p in model.parameters()]
            c_minus = [torch.zeros_like(p) for p in c_plus]

        scale = 1

        for (dataset, p_idx) in learned:
            task = tasks[dataset]
            i_perm = task.perms[0][p_idx]  # .cpu()
            if args.elasticity.signed_on_cpu:
                i_perm = i_perm.cpu()

            if task.perms[1] is None:
                t_perm = None
            else:
                t_perm = task.perms[1][p_idx]  # .cpu()
                if args.elasticity.signed_on_cpu:
                    t_perm = t_perm.cpu()

            used_no = 0

            loader: InMemoryDataLoader = task.train_loader
            old_bs, old_shuffle = loader.batch_size, loader.shuffle
            loader.batch_size, loader.shuffle = 1, False
            old_on_cuda = loader.is_cuda
            if args.elasticity.signed_on_cpu:
                loader.cpu()

            for data, target in loader:

                if self.do_sample and np.random.sample() > self.do_sample:
                    continue

                model.zero_grad()
                data, target = permute(data, target, i_perm, t_perm)
                output = model(Variable(data))

                # Keep all or just the correctly classified ones
                predicted_class = output.max(1, keepdim=False)[1][0]
                if self.drop_wrong and (predicted_class.data[0] != target[0]):
                    continue

                # Accumulate gradients
                loss, batch_used_no = self._loss(output, Variable(target))
                if batch_used_no < 1:
                    continue
                loss = loss * .001
                loss.backward()

                used_no += 1

                for t_i in zip(model.parameters(), c_plus, c_minus):
                    (p_i, cp_i, cm_i) = t_i
                    cp_i += (p_i.grad.data *
                             (p_i.grad.data > 0).float() * scale)
                    cm_i -= (p_i.grad.data *
                             (p_i.grad.data < 0).float() * scale)

                max_abs = max(max(p_i.abs().max() for p_i in c_plus),
                              max(p_i.abs().max() for p_i in c_plus))
                if max_abs > 10**3:
                    for p_i in c_plus:
                        p_i.div_(10)
                    for m_i in c_minus:
                        m_i.div_(10)
                    scale = .1 * scale
                    print(f"New scale = {scale:f}")

            self.used_no[(dataset, p_idx)] = used_no
            print(f"Used {used_no:d}/{len(task.train_loader.dataset):d} "
                  f"samples {(100.0 * used_no) / len(loader.dataset):f}% "
                  f"from {dataset:s}-{(p_idx+1):03d}.")

            if args.elasticity.signed_on_cpu and old_on_cuda:
                loader.cuda()

            loader.batch_size, loader.shuffle = old_bs, old_shuffle

        if args.cuda:
            model.cuda()
            c_plus = [t.cuda() for t in c_plus]
            c_minus = [t.cuda() for t in c_minus]
        self._coefficients = [[Variable(c) for c in c_plus],
                              [Variable(c) for c in c_minus]]

    def _loss(self, output: Variable, target: Variable) -> Tuple[Variable, int]:
        return F.cross_entropy(output, target), output.size(0)

    def __call__(self, model: nn.Module) -> float:
        losses = []  # type: List[Variable]
        is_signed = self.is_signed
        for _t in zip(model.parameters(), self.ref_params, *self.coefficients):
            if is_signed:
                p_t, p_0, c_p, c_m = _t
            else:
                p_t, p_0, c_t = _t

            diff = p_t - p_0  # type: Variable

            if is_signed:
                sgn = diff.sign()
                sgn_p, sgn_m = (sgn == 1).detach(), (sgn == -1).detach()

            if self.loss_norm == 2:
                diff = diff * diff
            else:
                diff = diff.abs()

            if is_signed:
                diff_p, diff_m = diff[sgn_p], diff[sgn_m]
                c_p, c_m = c_p[sgn_p], c_m[sgn_m]
                losses.append(torch.dot(c_p, diff_p) + torch.dot(c_m, diff_m))
            else:
                losses.append(torch.dot(c_t, diff))

        return sum(losses)

    @property
    def ref_params(self) -> Variables:
        return self.p_zero

    @property
    def coefficients(self) -> Coefficients:
        return [c for c in self._coefficients]


class APEC(ElasticConstraint):

    def __init__(self, model: nn.Module,
                 tasks: Tasks,
                 learned: List[Tuple[str, int]],
                 args: Args):
        print("Creating an action preserving elastic constraint"
              " around current parameters.")
        self.alpha = args.elasticity.alpha
        super(APEC, self).__init__(model, tasks, learned, args)

    def _loss(self, output: Variable, target: Variable) -> Optional[Variable]:
        probs = F.softmax(output, dim=1)
        q_max, _ = output.max(1, keepdim=True)
        cost = output - q_max.expand_as(output) + \
            (1 + self.alpha / probs).log()

        mask = (cost.data > 0)
        mask = mask & torch.ones_like(mask)\
                           .scatter_(1, target.data.unsqueeze(1), 0)

        if not mask.any():
            del probs, q_max, cost, mask
            return None, 0
        cost = cost[Variable(mask)]
        # for numerical stability (gradients tend to be very big)
        return torch.dot(cost, cost) * .01, (mask.sum(1) > 0).sum()


def elastic_loss(model: nn.Module,
                 tasks: Tasks,
                 learned: List[Tuple[str, int]],
                 args: Args) -> ElasticConstraint:
    if args.elasticity.mode == "ewc":
        return ElasticConstraint(model, tasks, learned, args)
    elif args.elasticity.mode == "apec":
        assert args.elasticity.drop_misclassified
        return APEC(model, tasks, learned, args)
    else:
        raise ValueError(args.mode)
