import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as functional
import torch.optim as optim
from typing import Union, Callable, Any, List, Dict, Iterator, Tuple, Type

# Project imports
from .base_agent import BaseAgent
from reporting import Reporting


class EWC(BaseAgent):
    def __init__(self, *args, **kwargs):
        super(EWC, self).__init__(*args, **kwargs)

        args = self._args
        agent_args = args.lifelong
        self.merge_elasticities = agent_args.merge_elasticities

        self.elasticity = dict({})
        self.elasticities = []
        self.weights = []

    def _train_task_batch(self, batch_idx: int, data: Tensor, targets: List[Tensor],
                          head_idx: Union[int, Tensor])-> Tuple[List[Tensor], Tensor, Dict]:

        model = self._model
        self._optimizer.zero_grad()
        outputs = model(data, head_idx=head_idx)
        task_no = self.crt_task_idx

        loss = torch.tensor(0., device=self.device)
        for out, t in zip(outputs, targets):
            loss += functional.cross_entropy(out, t)

        if task_no > 0:
            elasticity = self.elasticity
            elasticities = self.elasticities
            weights = self.weights

            loss_e = torch.tensor(0., device=self.device)

            if self.merge_elasticities:
                weight = weights[-1]
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        loss_e += torch.dot(elasticity[name].view(-1),
                                            (weight[name] - param).view(-1).pow(2))
            else:
                for ix, (weight, elasticity) in enumerate(zip(weights, elasticities)):
                    if ix == 0:
                        scale = 1000
                    else:
                        scale = 1

                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            if ix == 0:
                                loss_e += torch.dot(elasticity[name].view(-1),
                                                    (weight[name] - param).view(-1).pow(2).view(-1)) \
                                          * scale
                            else:
                                loss_e += torch.sum((weight[name] - param).view(-1).pow(2).view(
                                    -1)) * scale


            print(loss_e.item())
            loss += loss_e

        loss.backward()
        self._optimizer.step()

        return outputs, loss, dict({})

    def _end_train_task(self):
        train_loader, val_loader = self.crt_data_loaders
        assert hasattr(train_loader, "__len__")
        self._optimizer.zero_grad()
        model = self._model
        elasticity = self.elasticity

        for batch_idx, (data, targets, head_idx) in enumerate(train_loader):
            outputs = model(data, head_idx=head_idx)

            loss = torch.tensor(0., device=self.device)
            for out, t in zip(outputs, targets):
                loss += functional.cross_entropy(out, t)

            loss.backward()

        grad = dict({})
        weights = dict({})
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights[name] = param.detach().clone()
                grad[name] = param.grad.detach().clone()
                grad[name].pow_(2)
                if name in elasticity:
                    elasticity[name].add_(grad[name])
                else:
                    elasticity[name] = grad[name].clone()

        self.weights.append(weights)
        self.elasticities.append(grad)







