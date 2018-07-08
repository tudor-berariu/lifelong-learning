import torch
from torch import Tensor
import torch.nn.functional as functional
from typing import Union, List, Dict, Tuple, NamedTuple
from termcolor import colored as clr

# Project imports
from .base_agent import BaseAgent
from .ewc import EWC

Constraint = NamedTuple("Constraint", [("task_idx", int),
                                       ("epoch", int),
                                       ("mode", Dict[str, Tensor]),
                                       ("elasticity", Dict[str, Tensor])])


class KeepTaskData(EWC):
    def __init__(self, *args, **kwargs):
        super(EWC, self).__init__(*args, **kwargs)

        args = self._args
        agent_args = args.lifelong
        self.keep_cnt = agent_args.keep_cnt
        self.keep_train_sample = agent_args.keep_train_sample

    def _train_task_batch(self, batch_idx: int, data: Tensor, targets: List[Tensor],
                          head_idx: Union[int, Tensor])-> Tuple[List[Tensor], Tensor, Dict]:

        model = self._model
        self._optimizer.zero_grad()
        outputs = model(data, head_idx=head_idx)
        task_no = self.crt_task_idx

        loss = torch.tensor(0., device=self.device)
        for out, target in zip(outputs, targets):
            loss += functional.cross_entropy(out, target)

        # ==========================================================================================
        # -- FWD on data on older tasks
        older_tasks = 0
        sample_cnt = self.keep_train_sample
        for data_loader in enumerate(self.train_tasks):
            if older_tasks >= task_no or sample_cnt <= 0:
                break

            for o_batch_idx, (o_data, o_targets, o_head_idx) in enumerate(data_loader):
                o_data = o_data[:sample_cnt]
                o_targets = o_targets[:sample_cnt]
                if not isinstance(o_head_idx, int):
                    o_head_idx = o_head_idx[:sample_cnt]
                o_outputs = model(o_data, head_idx=o_head_idx)
                for o_out, o_target in zip(o_outputs, o_targets):
                    loss += functional.cross_entropy(o_out, o_target)
                sample_cnt -= o_data.size(0)

                if sample_cnt <= 0:
                    break

            older_tasks += 1
        # ==========================================================================================

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
