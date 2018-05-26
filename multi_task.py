from copy import copy
from typing import Iterator, List, NamedTuple, NewType, Optional, Tuple
from argparse import Namespace
from operator import mul
from functools import reduce

import torch
from torch import Tensor

from my_types import Args
from datasets import CLASSES_NO, InMemoryDataSet, DataSetFactory

Batch = Tuple[Tensor, Tensor]


class TaskDataLoader(object):

    def __init__(self,
                 name: str,
                 dataset: InMemoryDataSet,
                 in_perm: Optional[Tensor] = None,
                 out_perm: Optional[Tensor] = None,
                 batch_size: int = 0,
                 drop_last: bool = True,
                 shuffle: bool = False) -> None:
        self._name = name
        self.dataset = dataset
        self.__batch_size = batch_size if batch_size > 0 else len(dataset)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.in_perm = in_perm
        self.out_perm = out_perm
        self.__sh_idx = None
        self.__start_idx = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self.__batch_size = batch_size

    def to_(self, device) -> None:
        # Warning: dataset might be shared by multiple tasks
        self.dataset.to_(device)

    def __iter__(self) -> Iterator[Batch]:
        if self.shuffle and self.batch_size < len(self.dataset):
            self.__sh_idx = torch.randperm(len(self.dataset))
            self.__sh_idx = self.__sh_idx.to(self.dataset.data.device)
        else:
            self.__sh_idx = None
        self.__start_idx = 0
        return self

    def __next__(self) -> Batch:
        start_idx = self.__start_idx
        dataset = self.dataset
        set_size = len(dataset)
        if start_idx >= set_size:
            raise StopIteration
        end_idx = start_idx + self.__batch_size
        if end_idx > set_size:
            if self.drop_last:
                raise StopIteration
            else:
                end_idx = set_size
        if self.__sh_idx is not None:
            idxs = self.__sh_idx[start_idx:end_idx]
            data = dataset.data.index_select(0, idxs)
            target = dataset.target.index_select(0, idxs)
        else:
            data = dataset.data[start_idx:end_idx]
            target = dataset.data[start_idx:end_idx]
        self.__start_idx = end_idx
        return data, target

    def __len__(self):
        return len(self.dataset)


Task = NamedTuple("Task",
                  [("name", str),
                   ("train_set", InMemoryDataSet),
                   ("validation_set", Optional[InMemoryDataSet]),
                   ("test_set", InMemoryDataSet),
                   ("dataset_name", str),
                   ("classes", Optional[List[int]]),
                   ("p_idx", Optional[int]),
                   ("in_permutation", Optional[Tensor]),
                   ("out_permutation", Optional[Tensor])])


class MultiTask(object):

    def __init__(self, args: Args,
                 device: torch.device = torch.device("cpu")) -> None:

        datasets = args.tasks.datasets  # type: List[str]
        self._in_size = in_size = torch.Size(args.tasks.in_size)
        reset_targets = args.tasks.reset_targets  # type: bool
        validation = args.tasks.validation  # type: float
        split = args.tasks.split if args.tasks.split else 1
        perms_no = args.tasks.perms_no if args.tasks.perms_no else 1
        permute_targets = args.tasks.permute_targets  # type: bool

        self.batch_size = args.train.batch_size  # type: int
        self.shuffle = args.train.shuffle  # type: bool
        self.drop_last = False

        kwargs = {
            "reset_targets": reset_targets,
            "validation": validation,
            "device": device
        }
        in_n = reduce(mul, in_size, 1)  # type: int

        self._tasks = []
        self._out_size = []

        factory = DataSetFactory(datasets, in_size)

        for dataset_name in datasets:
            classes_no = CLASSES_NO[dataset_name]  # type: int
            if split > 1:
                all_classes = torch.randperm(classes_no).tolist()
                step = classes_no // split
                for start in range(0, classes_no, step):
                    end = min(classes_no, start + step)
                    classes = all_classes[start:end]
                    assert len(classes) > 1, "At least two classes, please."

                    kwargs["classes"] = classes
                    cls_name = ",".join([str(cls) for cls in classes])
                    trn_d, vld_d, tst_d = factory.get_datasets(
                        dataset_name, **kwargs)
                    if perms_no > 1:
                        for p_idx in range(perms_no):
                            in_perm = torch.randperm(in_n).to(device)
                            assert not args.task.permute_targets, "Don't!"
                            out_perm = None
                            name = f"{dataset_name:s}_c{cls_name:s}_p{p_idx:d}"
                            self._tasks.append(Task(name, trn_d, vld_d, tst_d,
                                                    dataset_name, classes,
                                                    p_idx, in_perm, out_perm))
                            self._out_size.append(len(classes))

                    else:
                        name = f"{dataset_name:s}_c{cls_name:s}"
                        self._tasks.append(Task(name, trn_d, vld_d, tst_d,
                                                dataset_name, classes,
                                                None, None, None))
                        self._out_size.append(len(classes))
            else:
                kwargs["classes"] = None
                trn_d, vld_d, tst_d = factory.get_datasets(
                    dataset_name, **kwargs)
                if perms_no > 1:
                    for p_idx in range(perms_no):
                        in_perm = torch.randperm(in_n).to(device)
                        if permute_targets:
                            out_perm = torch.randperm(classes_no)
                        else:
                            out_perm = None
                        name = f"{dataset_name:s}_p{p_idx:d}"
                        self._tasks.append(Task(name, trn_d, vld_d, tst_d,
                                                dataset_name, None,
                                                p_idx, in_perm, out_perm))
                        self._out_size.append(classes_no)

                else:
                    self._tasks.append(Task(dataset_name, trn_d, vld_d, tst_d,
                                            dataset_name, None, None, None, None))
                    self._out_size.append(classes_no)

    def __len__(self):
        return len(self._tasks)

    def train_tasks(self) -> Iterator[Tuple[TaskDataLoader, TaskDataLoader]]:
        for task in self._tasks:
            tr_dl = TaskDataLoader(task.name, task.train_set,
                                   task.in_permutation, task.out_permutation,
                                   self.batch_size, self.drop_last,
                                   self.shuffle)
            snd = task.validation_set if task.validation_set else task.test_set
            tst_dl = TaskDataLoader(task.name, snd,
                                    task.in_permutation, task.out_permutation,
                                    batch_size=0)
            yield (tr_dl, tst_dl)

    def test_tasks(self, first_n: int = 0) -> Iterator[TaskDataLoader]:
        for idx, task in enumerate(self._tasks):
            if idx >= first_n:
                break
            dst = task.validation_set if task.validation_set else task.test_set
            tst_dl = TaskDataLoader(task.name, dst,
                                    task.in_permutation, task.out_permutation,
                                    batch_size=0)
            yield tst_dl

    def get_task(self, idx: int) -> Tuple[TaskDataLoader, TaskDataLoader]:
        pass

    def merged_tasks(self) -> Tuple[TaskDataLoader, TaskDataLoader]:
        pass

    def task_names(self) -> List[str]:
        return [task.name for task in self._tasks]

    def tasks_info(self) -> List[Tuple[str, Optional[int], Optional[List[int]]]]:
        return [(task.name, task.p_idx, task.classes) for task in self._tasks]

    @property
    def out_size(self) -> List[int]:
        return copy(self._out_size)

    @property
    def in_size(self) -> torch.Size:
        return copy(self._in_size)


def test_split():
    args = Namespace(
        tasks=Namespace(
            datasets=["fashion", "mnist"],
            in_size=[3, 32, 32],
            reset_targets=False,
            validation=.8,
            split=2,
            perms_no=None,
            permute_targets=True
        ),
        train=Namespace(
            batch_size=6000,
            shuffle=True
        )
    )
    mt = MultiTask(args, torch.device("cuda:0"))
    for train_loader, valid_loader in mt.train_tasks():
        print(train_loader.name)
        for x, y in train_loader:
            print("new_batch")
        for x, y in valid_loader:
            print("new_batch")


def test_perms():
    args = Namespace(
        tasks=Namespace(
            datasets=["fashion", "mnist"],
            in_size=[3, 32, 32],
            reset_targets=False,
            validation=.8,
            split=False,
            perms_no=3,
            permute_targets=True
        ),
        train=Namespace(
            batch_size=6000,
            shuffle=True
        )
    )
    mt = MultiTask(args, torch.device("cuda:0"))
    for train_loader, valid_loader in mt.train_tasks():
        print(train_loader.name)
        for x, y in train_loader:
            print("new_batch")
        for x, y in valid_loader:
            print("new_batch")


if __name__ == "__main__":
    test_perms()
    test_split()
