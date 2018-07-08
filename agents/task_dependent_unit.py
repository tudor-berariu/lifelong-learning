import torch
from torch import Tensor
from typing import Union, List, Dict, Tuple, NamedTuple

# Project imports
from .ewc import EWC

Constraint = NamedTuple("Constraint", [("task_idx", int),
                                       ("epoch", int),
                                       ("mode", Dict[str, Tensor]),
                                       ("elasticity", Dict[str, Tensor])])


class TaskDependentUnitEWC(EWC):
    def __init__(self, *args, **kwargs):
        super(TaskDependentUnitEWC, self).__init__(*args, **kwargs)

        args = self._args
        agent_args = args.lifelong
        self.task_units = task_units = agent_args.task_units
        self.task_dependent_units = self.no_tasks * task_units
        self.masks = masks = {}
        all_mask = list(range(0, task_units * self.no_tasks))

        for tsk in range(self.no_tasks):
            mask_start = task_units * tsk
            mask_end = mask_start + task_units
            mask = all_mask[:mask_start] + all_mask[mask_end:]
            masks[tsk] = torch.LongTensor(mask).to(self.device)

        assert hasattr(self._model, "set_mask"), "Model doesn't have set_mask function"

    def _start_train_epoch(self):
        self.set_model_mask(self.crt_task_idx)

        for _idx, constraint in enumerate(self.constraints):
            for name, param in constraint.elasticity.items():
                param[-self.task_dependent_units:].mul_(0)

    def _start_eval_task(self):
        self.set_model_mask(self.crt_eval_task_idx)

    def set_model_mask(self, crt_task):
        self._model.set_mask(crt_task, self.masks[crt_task])



