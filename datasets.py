from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from multiprocessing.pool import ThreadPool


ORIGINAL_SIZE = {
    "mnist": torch.Size((1, 28, 28)),
    "fashion": torch.Size((1, 28, 28)),
    "cifar10": torch.Size((3, 32, 32)),
    "svhn": torch.Size((3, 32, 32)),
    "cifar100": torch.Size((3, 32, 32)),
    "fake":  torch.Size((3, 32, 32)),
}

MEAN_STD = {
    "mnist": {(3, 32, 32): (0.10003692801078261, 0.2752173485400458)},
    "fashion": {(3, 32, 32): (0.21899983604159193, 0.3318113789274)},
    "cifar10": {(3, 32, 32): (0.4733630111949825, 0.25156892869250536)},
    "cifar100": {(3, 32, 32): (0.478181,  0.268192)},
    "svhn": {(3, 32, 32): (0.451419, 0.199291)}
}

CLASSES_NO = {"mnist": 10, "fashion": 10, "cifar10": 10, "fake": 10,
              "svhn": 10, "cifar100": 100}

DATASETS = {
    "mnist": datasets.MNIST,
    "fashion": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "svhn": datasets.SVHN,
    "cifar100": datasets.CIFAR100
}


class InMemoryDataSet(object):

    def __init__(self,
                 data: Tensor, target: Tensor,
                 cut: Optional[Tuple[float, float]] = None,
                 classes: Optional[List[int]] = None,
                 reset_targets: bool = False) -> None:
        if cut:
            if not 0 <= cut[0] < cut[1] <= 1:
                raise ValueError
            length = data.size(0)
            start = round(length * cut[0])
            end = round(length * cut[1])
            data, target = data[start:end], target[start:end]

        if classes:
            idxs = (target == classes[0])
            for cls in classes[1:]:
                idxs = idxs | (target == cls)
            data, target = data[idxs], target[idxs]

            if reset_targets:
                mapping = {c: i for (i, c) in enumerate(classes)}
                target.apply_(lambda i: mapping[i])

        self.data, self.target = data, target

    def to_(self, device: torch.device) -> None:
        self.data, self.target = self.data.to(device), self.target.to(device)

    def __len__(self) -> int:
        return self.data.size(0)


Padding = Tuple[int, int, int, int]
Datasets = Tuple[InMemoryDataSet, Optional[InMemoryDataSet], InMemoryDataSet]


def get_padding(in_size: torch.Size, out_size: torch.Size) -> Padding:
    assert len(in_size) == len(out_size)
    d_h, d_w = out_size[-2] - in_size[-2], out_size[-1] - in_size[-1]
    p_h1, p_w1 = d_h // 2, d_w // 2
    p_h2, p_w2 = d_h - p_h1, d_w - p_w1
    return (p_h1, p_h2, p_w1, p_w2)


def get_mean_and_std(dataset_name: str,
                     in_size: torch.Size) -> Tuple[float, float]:
    if dataset_name in MEAN_STD:
        if tuple(in_size) in MEAN_STD[dataset_name]:
            return MEAN_STD[dataset_name][tuple(in_size)]

    original_size = ORIGINAL_SIZE[dataset_name]  # type: torch.Size
    padding = get_padding(original_size, in_size)  # type: Padding
    if dataset_name == "svhn":
        data = DATASETS[dataset_name](
            f'./.data/.{dataset_name:s}_data',
            split="train", download=True,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size))
            ]))
    else:
        data = DATASETS[dataset_name](
            f'./.data/.{dataset_name:s}_data',
            train=True, download=True,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size))
            ]))
    
    if hasattr(data, "train_data") and torch.is_tensor(data.train_data):
        batch_size = data.train_data.size(0)
    elif hasattr(data, "train_data") and isinstance(data.train_data, np.ndarray):
        batch_size = data.train_data.shape[0]
    else:
        batch_size = data.data.shape[0]

    loader = DataLoader(data, batch_size=batch_size)
    full_data, _ = next(iter(loader))  # get dataset in one batch
    mean, std = full_data.mean(), full_data.std()
    del loader, full_data

    print(f"Mean and std for {dataset_name:s} and {tuple(in_size)} are"
          f"{mean: 8.6f}, {std: 8.6f}.")

    return mean, std


def load_data_async(dataset_name: str,
                    in_size: Optional[torch.Size] = None):

    original_size = ORIGINAL_SIZE[dataset_name]
    in_size = in_size if in_size is not None else original_size

    if dataset_name == "fake":
        return torch.randn(20000, *in_size), \
            torch.LongTensor(20000).random_(10), \
            torch.randn(2000, *in_size), \
            torch.LongTensor(2000).random_(10), \


    padding = get_padding(original_size, in_size)
    mean, std = get_mean_and_std(dataset_name, in_size)

    if dataset_name == "svhn":
        train_data = DATASETS[dataset_name](
            f'./.data/.{dataset_name:s}_data',
            split="train", download=True,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size)),
                transforms.Normalize((mean,), (std,))
            ]))
    else:
        train_data = DATASETS[dataset_name](
            f'./.data/.{dataset_name:s}_data',
            train=True, download=True,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size)),
                transforms.Normalize((mean,), (std,))
            ]))    
    if dataset_name == "svhn":
        test_data = DATASETS[dataset_name](
            f'./.data/.{dataset_name:s}_data',
            split="test", download=True,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size)),
                transforms.Normalize((mean,), (std,))
            ]))
    else:
        test_data = DATASETS[dataset_name](
            f'./.data/.{dataset_name:s}_data',
            train=False, download=True,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size)),
                transforms.Normalize((mean,), (std,))
            ]))

    loader = DataLoader(train_data, batch_size=len(train_data),
                        num_workers=4)
    train_data, train_target = next(iter(loader))
    del loader

    loader = DataLoader(test_data, batch_size=len(test_data),
                        num_workers=4)
    test_data, test_target = next(iter(loader))
    del loader

    return train_data, train_target, test_data, test_target


class DataSetFactory(object):

    def __init__(self, all_datasets: List[str],
                 in_size: Optional[torch.Size] = None) -> None:
        self.full_data = {}
        pool = ThreadPool(processes=len(all_datasets))
        for dataset_name in all_datasets:
            self.full_data[dataset_name] = pool.apply_async(
                load_data_async, (dataset_name, in_size))

    def get_datasets(self,
                     dataset_name: str,
                     classes: Optional[List[int]]=None,
                     reset_targets: bool=False,
                     validation: Optional[float]=.1,
                     device: torch.device=torch.device("cpu")) -> Datasets:

        train_data, train_target, test_data, test_target = \
            self.full_data[dataset_name].get()

        kwargs = {"classes": classes, "reset_targets": reset_targets}

        if validation:
            if not .0 < validation <= 1:
                raise ValueError

            t_cut = (0, 1. - validation)
            v_cut = (1. - validation, 1.)

            train_set = InMemoryDataSet(
                train_data, train_target, cut=t_cut, **kwargs)
            valid_set = InMemoryDataSet(
                train_data, train_target, cut=v_cut, **kwargs)

        else:
            train_set = InMemoryDataSet(train_data, train_target, **kwargs)
            valid_set = None

        test_set = InMemoryDataSet(test_data, test_target, **kwargs)

        for data_set in [train_set, valid_set, test_set]:
            if data_set is not None:
                data_set.to_(device)

        return train_set, valid_set, test_set
