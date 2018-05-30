from copy import copy
from typing import Iterator, List, NamedTuple, Optional, Tuple, Union
from argparse import Namespace
from operator import mul
from functools import reduce
import numpy as np
import torch
from torch import Tensor

from my_types import Args
from datasets import CLASSES_NO, InMemoryDataSet, DataSetFactory

Batch = Tuple[Tensor, List[Tensor], Union[int, Tensor]]


def permute(batch: Tensor, permutation: Tensor) -> Tensor:
    size = batch.size()
    return batch.view(size[0], -1).index_select(1, permutation).view(size)


def permute_targets(batch: Tensor, permutation: Tensor):
    return permutation[batch]


Task = NamedTuple("Task",
                  [("name", str),
                   ("train_set", InMemoryDataSet),
                   ("validation_set", Optional[InMemoryDataSet]),
                   ("test_set", InMemoryDataSet),
                   ("dataset_name", str),
                   ("classes", Optional[List[int]]),
                   ("p_idx", Optional[int]),
                   ("in_permutation", Optional[Tensor]),
                   ("out_permutation", Optional[Tensor]),
                   ("head_idx", int)])


class TaskDataLoader(object):

    def __init__(self,
                 task: Task,
                 part: str = "train",
                 batch_size: int = 0,
                 drop_last: bool = False,
                 shuffle: bool = False) -> None:

        self._task = task
        if part == "train":
            self.dataset = task.train_set
        elif part == "validation":
            self.dataset = task.validation_set
        elif part == "test":
            self.dataset = task.test_set
        self.__batch_size = batch_size if batch_size > 0 else len(self.dataset)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.__sh_idx = None
        self.__start_idx = None

    @property
    def name(self) -> str:
        return self._task.name

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self.__batch_size = batch_size

    def to_(self, device) -> None:
        # Warning: dataset might be shared by multiple tasks
        self.dataset.to_(device)
        if self._task.in_permutation:
            self._task.in_permutation = self._task.in_permutation.to(device)
        if self._task.out_permutation:
            self._task.out_permutation = self._task.out_permutation.to(device)

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
            target = dataset.target[start_idx:end_idx]
        in_perm, out_perm = self._task.in_permutation, self._task.out_permutation
        if in_perm is not None:
            data = permute(data, in_perm)
        if out_perm is not None:
            target = permute_targets(target, out_perm)

        self.__start_idx = end_idx
        return data, [target], self._task.head_idx

    def __len__(self):
        return len(self.dataset)


class MultiTask(object):

    def __init__(self, args: Args) -> None:

        device: torch.device = torch.device(args.device)
        datasets = args.tasks.datasets  # type: List[str]
        self._in_size = in_size = torch.Size(args.tasks.in_size)
        reset_targets = args.tasks.reset_targets  # type: bool
        validation = args.tasks.validation  # type: float
        split = args.tasks.split if args.tasks.split else 1
        perms_no = args.tasks.perms_no if args.tasks.perms_no else 1
        permute_targets = args.tasks.permute_targets  # type: bool
        self.common_head = common_head = args.tasks.common_head  # type: bool

        self.batch_size = args.train.batch_size  # type: int
        self.test_batch_size = args.train.test_batch_size  # type: int
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

        head_idx = 0
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
                            assert not args.tasks.permute_targets, "Don't!"
                            out_perm = None
                            name = f"{dataset_name:s}_c{cls_name:s}_p{p_idx:d}"
                            self._tasks.append(Task(name, trn_d, vld_d, tst_d, dataset_name, classes,
                                                    p_idx, in_perm, out_perm, head_idx))
                            self._out_size.append(len(classes))
                            if not common_head:
                                head_idx += 1
                        if not common_head:
                            head_idx -= 1

                    else:
                        name = f"{dataset_name:s}_c{cls_name:s}"
                        self._tasks.append(Task(name, trn_d, vld_d, tst_d, dataset_name, classes,
                                                None, None, None, head_idx))
                        self._out_size.append(len(classes))
            else:
                kwargs["classes"] = None
                trn_d, vld_d, tst_d = factory.get_datasets(
                    dataset_name, **kwargs)
                if perms_no > 1:
                    for p_idx in range(perms_no):
                        in_perm = torch.randperm(in_n).to(device)
                        if permute_targets:
                            out_perm = torch.randperm(classes_no).to(device)
                        else:
                            out_perm = None
                        name = f"{dataset_name:s}_p{p_idx:d}"
                        self._tasks.append(Task(name, trn_d, vld_d, tst_d, dataset_name, None,
                                                p_idx, in_perm, out_perm, head_idx))
                        self._out_size.append(classes_no)

                        if not common_head:
                            head_idx += 1
                    if not common_head:
                        head_idx -= 1

                else:
                    self._tasks.append(Task(dataset_name, trn_d, vld_d, tst_d, dataset_name, None,
                                            None, None, None, head_idx))
                    self._out_size.append(classes_no)
            if not common_head:
                head_idx += 1

    def __len__(self):
        return len(self._tasks)

    def train_tasks(self) -> Iterator[Tuple[TaskDataLoader, TaskDataLoader]]:
        for task in self._tasks:
            train_dataloader = TaskDataLoader(task, "train", batch_size=self.batch_size,
                                              drop_last=self.drop_last, shuffle=self.shuffle)
            snd_part = "validation" if task.validation_set else "test"
            test_dataloader = TaskDataLoader(
                task, snd_part, batch_size=self.test_batch_size)
            yield train_dataloader, test_dataloader

    def test_tasks(self, first_n: int = 0) -> Iterator[TaskDataLoader]:
        for idx, task in enumerate(self._tasks):
            if idx >= first_n:
                break
            snd_part = "validation" if task.validation_set else "test"
            test_dataloader = TaskDataLoader(
                task, snd_part, batch_size=self.test_batch_size)
            yield test_dataloader

    def get_task_info(self):
        task: Task
        tasks_info = []
        for idx, task in enumerate(self._tasks):
            tasks_info.append({"idx": idx,
                               "dataset_name": task.dataset_name,
                               "classes": task.classes,
                               "p_idx": task.p_idx,
                               "name": task.name,
                               "best_individual": 1.,
                               "best_simultaneous": 1.,
                               "head": task.head_idx})
        return tasks_info

    def merged_tasks(self) -> Iterator[Batch]:
        batch_size = self.batch_size // len(self)
        loaders = []
        kwargs = {"drop_last": self.drop_last, "batch_size": batch_size, "shuffle": self.shuffle}

        for task in self._tasks:
            loaders.append(iter(TaskDataLoader(task, "train", **kwargs)))

        while True:
            inputs: List[Tensor] = []
            targets: List[Tensor] = []
            heads: List[Tensor] = []

            new_loaders = []
            for loader, task in zip(loaders, self._tasks):
                try:
                    (data, target, head) = next(loader)
                    new_loaders.append(loader)
                except StopIteration:
                    new_iter = iter(TaskDataLoader(task, "train", **kwargs))
                    (data, target, head) = next(new_iter)
                    new_loaders.append(new_iter)
                inputs.append(data)
                targets.extend(target)
                heads.extend([t.clone().fill_(head) for t in target])

            full_data: Tensor
            full_target: List[Tensor]
            full_heads: Tensor

            if len(inputs) > 1:
                full_data = torch.cat(inputs, dim=0)
                if self.common_head:
                    full_target = [torch.cat(targets, dim=0)]
                    full_heads = full_target[0].clone().fill_(0)
                else:
                    full_target = targets
                    full_heads = torch.cat(heads, dim=0)
            else:
                full_data, full_target, full_heads = inputs[0], targets, heads[0]

            yield full_data, full_target, full_heads
            loaders = new_loaders

    @property
    def average_batches_per_epoch(self):
        sizes = []
        for task in self._tasks:
            sizes.append(len(task.train_set))
        batches_cnt = np.mean(sizes) // (self.batch_size // len(self))
        return batches_cnt

    def task_names(self) -> List[str]:
        return [task.name for task in self._tasks]

    def tasks_info(self) -> List[Tuple[str, Optional[int], Optional[List[int]]]]:
        return [(task.name, task.p_idx, task.classes) for task in self._tasks]

    @property
    def out_size(self) -> List[int]:
        return [max(self._out_size)] if self.common_head else copy(self._out_size)

    @property
    def in_size(self) -> torch.Size:
        return copy(self._in_size)


# ---------------------------------------------------------------------------
# Testing

def test_split():
    args = Namespace(
        tasks=Namespace(
            datasets=["fake"],
            in_size=[3, 32, 32],
            reset_targets=False,
            validation=.8,
            split=2,
            perms_no=None,
            permute_targets=True,
            common_head=True
        ),
        device=torch.device("cuda:0"),
        train=Namespace(
            batch_size=6000,
            test_batch_size=5000,
            shuffle=True
        )
    )
    multi_task = MultiTask(args)
    for train_loader, valid_loader in multi_task.train_tasks():
        print(train_loader.name)
        for _inputs, _targets, _heads in train_loader:
            pass
        for _inputs, _targets, _heads in valid_loader:
            pass


def test_perms():
    args = Namespace(
        tasks=Namespace(
            datasets=["fake"],
            in_size=[3, 32, 32],
            reset_targets=False,
            validation=.8,
            split=False,
            perms_no=3,
            permute_targets=True,
            common_head=False
        ),
        device=torch.device("cuda:0"),
        train=Namespace(
            batch_size=6000,
            test_batch_size=5000,
            shuffle=True
        )
    )
    multi_task = MultiTask(args)
    for train_loader, valid_loader in multi_task.train_tasks():
        for _train_loader, _valid_loader, _heads in train_loader:
            pass
        for _train_loader, _valid_loader, _heads in valid_loader:
            pass


def test_simul():
    args = Namespace(
        tasks=Namespace(
            datasets=["fake"],
            in_size=[3, 32, 32],
            reset_targets=False,
            validation=.8,
            split=False,
            perms_no=3,
            permute_targets=True,
            common_head=True

        ),
        device=torch.device("cuda:0"),
        train=Namespace(
            batch_size=6000,
            test_batch_size=5000,
            shuffle=True
        )
    )
    multi_task = MultiTask(args)
    i = 0
    for _inputs, _targets, _heads in multi_task.merged_tasks():
        i += 1
        if i > 2 * multi_task.average_batches_per_epoch:
            break


if __name__ == "__main__":
    test_simul()
    test_perms()
    test_split()
