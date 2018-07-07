import sys
import time
from typing import Union, List, Dict, Tuple, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as functional
from torch.distributions import Categorical
from termcolor import colored as clr

# Project imports
from .base_agent import BaseAgent

Constraint = NamedTuple("Constraint", [("task_idx", int),
                                       ("epoch", int),
                                       ("mode", Dict[str, Tensor]),
                                       ("elasticity", Dict[str, Tensor])])


class FIMEWC(BaseAgent):
    def __init__(self, *args, **kwargs):
        super(FIMEWC, self).__init__(*args, **kwargs)

        args = self._args
        agent_args = args.lifelong
        self.merge_elasticities = agent_args.merge_elasticities

        self.scale = agent_args.scale
        self.samples_no = agent_args.samples_no
        self.empirical = agent_args.empirical
        self.verbose = agent_args.verbose

        self.saved_tasks_no = 0
        self.constraints: List[Constraint] = []

    def _train_task_batch(self, batch_idx: int, data: Tensor, targets: List[Tensor],
                          head_idx: Union[int, Tensor])-> Tuple[List[Tensor], Tensor, Dict]:

        model = self._model
        self._optimizer.zero_grad()
        outputs = model(data, head_idx=head_idx)
        task_no = self.crt_task_idx

        loss = torch.tensor(0., device=self.device)
        for out, target in zip(outputs, targets):
            loss += functional.cross_entropy(out, target)

        total_elastic_loss = torch.tensor(0., device=self.device)
        loss_per_layer = dict({})
        if task_no > 0:
            for _idx, constraint in enumerate(self.constraints):
                for name, param in model.named_parameters():
                    if param.grad is not None and name in constraint.elasticity:
                        loss_name = "loss_" + name
                        layer_loss = torch.dot(constraint.elasticity[name],
                                               (constraint.mode[name] - param.view(-1)).pow(2))
                        loss_per_layer[loss_name] = layer_loss.item() + \
                            loss_per_layer.get(loss_name, 0)
                        total_elastic_loss += layer_loss

            loss += total_elastic_loss * self.scale
            loss_per_layer["loss_ewc"] = total_elastic_loss.item()

        loss.backward()
        self._optimizer.step()

        return outputs, loss, loss_per_layer

    def _end_train_task(self):

        if self.crt_task_idx > 1 and self.first_task_only:
            return

        train_loader, _ = self.crt_data_loaders
        samples_no = self.samples_no
        model = self._model
        empirical = self.empirical
        verbose = self.verbose

        fim = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fim[name] = torch.zeros_like(param)

        seen_no = 0
        last = 0
        tic = time.time()

        train_iterator = iter(train_loader)
        while samples_no is None or seen_no < samples_no:
            try:
                (data, targets, head_idx) = next(train_iterator)
            except StopIteration:
                if samples_no is None:
                    break
                train_iterator = iter(train_loader)
                (data, targets, head_idx) = next(train_iterator)

            logits = model(data, head_idx=head_idx)
            assert isinstance(logits, list) and len(logits) == 1
            logits = functional.log_softmax(logits[0], dim=1)

            if empirical:
                outdx = targets[0].unsqueeze(1)
            else:
                outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
            samples = logits.gather(1, outdx)
            idx, batch_size = 0, data.size(0)
            while idx < batch_size and (samples_no is None or seen_no < samples_no):
                model.zero_grad()
                torch.autograd.backward(samples[idx], retain_graph=True)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        fim[name] += (param.grad * param.grad)
                        fim[name].detach_()
                seen_no += 1
                idx += 1

            if verbose and seen_no % 100 == 0:
                toc = time.time()
                fps = float(seen_no - last) / (toc - tic)
                tic, last = toc, seen_no
                sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.4f} samples/s.")

        if verbose:
            if seen_no > last:
                toc = time.time()
                fps = float(seen_no - last) / (toc - tic)
            sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.5f} samples/s.\n")

        for name, grad2 in fim.items():
            grad2.div_(float(seen_no))

        if self.merge_elasticities and self.constraints:
            # Add to previous matrices if in `merge` mode
            elasticity = self.constraints[0].elasticity
        else:
            elasticity = dict({})

        crt_mode = dict({})
        for name, param in model.named_parameters():
            if param.grad is not None:
                crt_mode[name] = param.detach().clone().view(-1)
                if name in elasticity:
                    elasticity[name].add_(fim[name].view(-1)).view(-1)
                else:
                    elasticity[name] = fim[name].view(-1)

        new_constraint = Constraint(self.crt_task_idx, self.crt_task_epoch, crt_mode, elasticity)
        if self.merge_elasticities:
            # Remove old constraints if in `merge` mode
            self.constraints.clear()
        self.constraints.append(new_constraint)

        print(clr(f"There are {len(self.constraints):d} elastic constraints!", attrs=["bold"]))
