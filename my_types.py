from typing import Dict, NamedTuple, Optional, Tuple, TypeVar, Union
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import Tensor

from torchutils import CudaDataLoader

# Types used in this program

Args = Namespace
Loader = Union[DataLoader, CudaDataLoader]
Loaders = Tuple[Loader, Loader]

LongVector = TypeVar('LongVector')
LongMatrix = TypeVar('LongMatrix')
Permutations = Tuple[LongMatrix, Optional[LongMatrix]]
DatasetTasks = NamedTuple("DatasetTasks", [("train_loader", DataLoader),
                                           ("test_loader", DataLoader),
                                           ("perms", Permutations)])
Tasks = Dict[str, DatasetTasks]
Model = Module


__all__ = ["Args", "Loaders",
           "Tensor",
           "LongVector", "LongMatrix",
           "Permutations", "DatasetTasks", "Tasks",
           "Model"]
