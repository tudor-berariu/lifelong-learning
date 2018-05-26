from typing import Iterator, List, Tuple
from argparse import Namespace

from torch import Tensor

Args = Namespace
Batch = Tuple[Tensor, Tensor]


class DataLoader(object):

    def __init__(self) -> None:
        pass

    @property
    def dataset_name(self) -> str:
        pass

    @property
    def permutation_idx(self) -> int:
        pass

    @property
    def classes(self) -> List[int]:
        pass

    @property
    def batch_size(self) -> int:
        pass

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        pass

    def cpu(self) -> None:
        pass

    def cuda(self) -> None:
        pass

    @property
    def is_cuda(self) -> bool:
        pass

    def __iter__(self) -> Iterator[Batch]:
        pass

    def __next__(self) -> Batch:
        pass

    def __length__(self):
        pass


class MultiTask(object):

    def __init__(self, args: Namespace) -> None:
        pass

    def __length__(self):
        pass

    def train_tasks(self) -> Iterator[DataLoader]:
        pass

    def test_tasks(self, first_n: int = 0) -> Iterator[DataLoader]:
        pass

    def get_task(self, idx: int) -> Tuple[DataLoader, DataLoader]:
        pass

    def merged_tasks(self) -> Tuple[DataLoader, DataLoader]:
        pass

    def task_names(self) -> List[str]:
        pass

    def task_info(self) -> List[Tuple[str, int, List[int]]]:
        pass
