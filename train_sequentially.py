import torch
import torch.nn as nn
from typing import Type, Callable

# Project imports
from my_types import Args, Tasks, Model, LongVector, DatasetTasks


def train_sequentially(model_class: Type,
                       get_optimizer: Callable[nn.Module],
                       tasks: Tasks,
                       args: Args)-> None:

    return None