import torch
import torch.nn as nn
import torch.optim as optim
from typing import Type, Callable

# Project imports
from my_types import Args, Tasks, Model, LongVector, DatasetTasks
from multi_task import MultiTask
from reporting import Reporting


def train_sequentially(model_class: Type,
                       get_optimizer: Callable[[nn.Module], optim.Optimizer],
                       multitask: MultiTask,
                       args: Args)-> None:

    return None