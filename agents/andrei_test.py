import torch
from torch import Tensor
import torch.nn.functional as functional
from typing import Union, List, Dict, Tuple, NamedTuple
from termcolor import colored as clr

# Project imports
from .base_agent import BaseAgent

Constraint = NamedTuple("Constraint", [("task_idx", int),
                                       ("epoch", int),
                                       ("mode", Dict[str, Tensor]),
                                       ("elasticity", Dict[str, Tensor])])


class AndreiTest(BaseAgent):
    def __init__(self, *args, **kwargs):
        super(AndreiTest, self).__init__(*args, **kwargs)

        args = self._args
        agent_args = args.lifelong
        self.merge_elasticities = agent_args.merge_elasticities

        self.first_task_only = agent_args.first_task_only
        self.scale = agent_args.scale
        self.saved_tasks_no = 0

        self.constraints: List[Constraint] = []

    def _train_task_batch(self, batch_idx: int, data: Tensor, targets: List[Tensor],
                          head_idx: Union[int, Tensor]) -> Tuple[List[Tensor], Tensor, Dict]:

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
        assert hasattr(train_loader, "__len__")
        self._optimizer.zero_grad()
        model = self._model

        for _batch_idx, (data, targets, head_idx) in enumerate(train_loader):
            outputs = model(data, head_idx=head_idx)
            loss = torch.tensor(0., device=self.device)
            for output, target in zip(outputs, targets):
                loss += functional.cross_entropy(output, target)
            loss.backward()

        grad = dict({})
        crt_mode = dict({})

        if self.merge_elasticities and self.constraints:
            # Add to previous matrices if in `merge` mode
            elasticity = self.constraints[0].elasticity
        else:
            elasticity = dict({})

        for name, param in model.named_parameters():
            if param.grad is not None:
                crt_mode[name] = param.detach().clone().view(-1)
                grad[name] = param.grad.detach().pow(2).clone().view(-1)
                if name in elasticity:
                    elasticity[name].add_(grad[name]).view(-1)
                else:
                    elasticity[name] = grad[name].clone().view(-1)

        new_constraint = Constraint(self.crt_task_idx, self.crt_task_epoch, crt_mode, elasticity)
        if self.merge_elasticities:
            # Remove old constraints if in `merge` mode
            self.constraints.clear()
        self.constraints.append(new_constraint)

        print(clr(f"There are {len(self.constraints):d} elastic constraints!", attrs=["bold"]))
