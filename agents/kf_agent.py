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
from models.kfac import GaussianPrior

class KFAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super(KFAgent, self).__init__(*args, **kwargs)

        args = self._args
        agent_args = args.lifelong
        self.merge_elasticities = agent_args.merge_elasticities
        self.first_task_only = False
        self.scale = agent_args.scale
        self.samples_no = agent_args.samples_no
        self.empirical = agent_args.empirical
        self.tridiag = agent_args.tridiag
        self.average_factors = agent_args.average_factors
        self.use_batches = agent_args.use_batches

        if agent_args.only_correct:
            raise NotImplementedError

        if self.empirical:
            self.only_correct = agent_args.only_correct
        self.verbose = agent_args.verbose

        self.saved_tasks_no = 0
        self.constraints: List[GaussianPrior] = []

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
        use_batches = self.use_batches

        model.average_factors = self.average_factors
        model.tridiag = self.tridiag

        model.start_kf()

        seen_no = 0
        last = 0
        tic = time.time()

        train_iterator = iter(train_loader)
        while not samples_no or seen_no < samples_no:
            try:
                (data, targets, head_idx) = next(train_iterator)
            except StopIteration:
                if not samples_no:
                    break
                train_iterator = iter(train_loader)
                (data, targets, head_idx) = next(train_iterator)

            if use_batches:
                outputs = model(data, head_idx=head_idx)
                assert isinstance(outputs, list) and len(outputs) == 1
                logits = functional.log_softmax(outputs[0], dim=1)
                if empirical:
                    outdx = target.unsqueeze(1)
                else:
                    outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
                samples = logits.gather(1, outdx)
                model.zero_grad()
                torch.autograd.backward(samples.mean(), retain_graph=True)
                seen_no += samples.size(0)

                if verbose and seen_no - last >= 100:
                    toc = time.time()
                    fps = float(seen_no - last) / (toc - tic)
                    tic, last = toc, seen_no
                    sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.4f} samples/s.")
            else:
                idx, batch_size = 0, data.size(0)
                while idx < batch_size and (not samples_no or seen_no < samples_no):
                    outputs = model(data, head_idx=head_idx)
                    assert isinstance(outputs, list) and len(outputs) == 1
                    logits = functional.log_softmax(outputs[0], dim=1)
                    if empirical:
                        outdx = target[idx:idx+1].unsqueeze(1)
                    else:
                        outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
                    sample = logits.gather(1, outdx)
                    model.zero_grad()
                    torch.autograd.backward(sample, retain_graph=True)
                    seen_no += 1
                    idx += 1

                if verbose and seen_no % 1000 == 0:
                    toc = time.time()
                    fps = float(seen_no - last) / (toc - tic)
                    tic, last = toc, seen_no
                    sys.stdout.write(f"\rSamples: {seen_no:5d}."
                                     f" Fps: {fps:2.4f} samples/s."
                                     f" Batch size: {batch_size:d}.")
        if verbose:
            if seen_no > last:
                toc = time.time()
                fps = float(seen_no - last) / (toc - tic)
            sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.5f} samples/s.\n")

        gaussian_prior = model.end_kf()
        self.constraints.append(gaussian_prior)

        print(clr(f"There are {len(self.constraints):d} elastic constraints!", attrs=["bold"]))
