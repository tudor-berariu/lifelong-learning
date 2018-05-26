from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ORIGINAL_SIZE = {
    "mnist": torch.Size((1, 28, 28)),
    "fashion": torch.Size((1, 28, 28)),
    "cifar10": torch.Size((3, 32, 32))
}

MEAN_STD = {
    "mnist": {(3, 32, 32): (0.10003692801078261, 0.2752173485400458)},
    "fashion": {(3, 32, 32): (0.21899983604159193, 0.3318113789274)},
    "cifar10": {(3, 32, 32): (0.4733630111949825, 0.25156892869250536)}
}

DATASETS = {
    "mnist": datasets.MNIST,
    "fashion": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10
}


class InMemoryDataSet(object):

    def __init__(self,
                 dataset: Dataset,
                 cut: Optional[Tuple[float, float]] = None,
                 classes: Optional[List[int]] = None,
                 reset_targets: bool = False) -> None:

        loader = DataLoader(dataset, batch_size=len(dataset))
        data, target = next(iter(loader))
        del loader

        if cut:
            if not 0 <= cut[0] < cut[1] < 1:
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

    def __length__(self) -> int:
        return self.data.size(0)


Padding = Tuple[int, int, int, int]


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
    data = DATASETS[dataset_name](
        f'./.data/.{dataset_name:s}_data',
        train=True, download=True,
        transform=transforms.Compose([
            transforms.Pad(padding),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.expand(in_size))
        ]))
    if torch.is_tensor(data.train_data):
        batch_size = data.train_data.size(0)
    elif isinstance(data.train_data, np.ndarray):
        batch_size = data.train_data.shape[0]

    loader = DataLoader(data, batch_size=batch_size)
    full_data, _ = next(iter(loader))  # get dataset in one batch
    mean, std = full_data.mean(), full_data.std()
    del loader, full_data

    print(f"Mean and std for {dataset_name:s} and {tuple(in_size)} are"
          f"{mean: 8.6f}, {std: 8.6f}.")

    return mean, std


Datasets = Tuple[InMemoryDataSet, InMemoryDataSet]


def get_datasets(dataset_name: str,
                 in_size: torch.Size,
                 classes: Optional[List[int]] = None,
                 validation: Optional[float] = .1,
                 reset_targets: bool = False,
                 device: torch.device = torch.device("cpu")) -> Datasets:

    original_size = ORIGINAL_SIZE[dataset_name]
    padding = get_padding(original_size, in_size)
    mean, std = get_mean_and_std(dataset_name, in_size)

    train_data = DATASETS[dataset_name](
        f'./.data/.{dataset_name:s}_data',
        train=True, download=True,
        transform=transforms.Compose([
            transforms.Pad(padding),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.expand(in_size)),
            transforms.Normalize((mean,), (std,))
        ]))

    test_data = DATASETS[dataset_name](
        f'./.data/.{dataset_name:s}_data',
        train=False, download=True,
        transform=transforms.Compose([
            transforms.Pad(padding),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.expand(in_size)),
            transforms.Normalize((mean,), (std,))
        ]))

    train_data.to_(device)
    test_data.to_(device)

    kwargs = {"classes": classes, "reset_targets": reset_targets}

    if validation:
        if not .0 < validation <= 1:
            raise ValueError

        t_cut = (0, 1. - validation)
        v_cut = (1. - validation, 1.)

        train_set = InMemoryDataSet(train_data, cut=t_cut, **kwargs)
        valid_set = InMemoryDataSet(train_data, cut=v_cut, **kwargs)

    else:
        train_set = InMemoryDataSet(train_data, **kwargs)
        valid_set = None

    test_set = InMemoryDataSet(test_data, **kwargs)

    return train_set, valid_set, test_set
