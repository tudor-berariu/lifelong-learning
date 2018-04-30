from typing import Dict, NamedTuple, Optional, Tuple, TypeVar
from argparse import Namespace
from torch.nn import Module

from torchutils import InMemoryDataLoader

# Types used in this program

Args = Namespace
Loaders = Tuple[InMemoryDataLoader, InMemoryDataLoader]

LongVector = TypeVar('LongVector')
LongMatrix = TypeVar('LongMatrix')
Permutations = Tuple[LongMatrix, Optional[LongMatrix]]
DatasetTasks = NamedTuple("DatasetTasks",
                          [("train_loader", InMemoryDataLoader),
                           ("test_loader", InMemoryDataLoader),
                           ("perms", Permutations)])
Tasks = Dict[str, DatasetTasks]
Model = Module


__all__ = ["Args", "Loaders", "LongVector", "LongMatrix",
           "Permutations", "DatasetTasks", "Tasks",
           "Model"]
