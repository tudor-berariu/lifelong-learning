from typing import Optional, Tuple
from functools import reduce
from operator import mul

# Torch imports
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Types used across modules
from my_types import Args, Loaders, Permutations, DatasetTasks, Tasks,\
    LongVector, LongMatrix

from liftoff.config import value_of
from torchutils import CudaDataLoader


Padding = Tuple[int, int, int, int]

# Constants

DATASETS = {
    "mnist": datasets.MNIST,
    "fashion": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10
}

ORIGINAL_SIZE = {
    "mnist": torch.Size((1, 28, 28)),
    "fashion": torch.Size((1, 28, 28)),
    "cifar10": torch.Size((3, 32, 32))
}

CLASSES_NO = {"mnist": 10, "fashion": 10, "cifar10": 10}

MEAN_STD = {
    "mnist": {(3, 32, 32): (0.10003692801078261, 0.2752173485400458)},
    "fashion": {(3, 32, 32): (0.21899983604159193, 0.3318113789274)},
    "cifar10": {(3, 32, 32): (0.4733630111949825, 0.25156892869250536)}
}


def get_padding(in_size: torch.Size, out_size: torch.Size) -> Padding:
    assert len(in_size) == len(out_size)
    d_h, d_w = out_size[-2] - in_size[-2], out_size[-1] - in_size[-1]
    p_h1, p_w1 = d_h // 2, d_w // 2
    p_h2, p_w2 = d_h - p_h1, d_w - p_w1
    return (p_h1, p_h2, p_w1, p_w2)


def get_mean_and_std(dataset: str, args: Args) -> Tuple[float, float]:
    in_size = args.in_size  # type: torch.Size

    if dataset in MEAN_STD and tuple(in_size) in MEAN_STD[dataset]:
        return MEAN_STD[dataset][tuple(in_size)]

    original_size = ORIGINAL_SIZE[dataset]  # type: torch.Size
    padding = get_padding(original_size, in_size)  # type: Padding
    data = DATASETS[dataset](
        f'./.data/.{dataset:s}_data',
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

    print(f"Mean and std for {dataset} and {tuple(in_size)} are", mean, std)

    return mean, std


def get_loaders(dataset: str, batch_size: int, args: Args) -> Loaders:
    if args.cuda:
        tLoader = CudaDataLoader
        kwargs = {}
    else:
        tLLoader = DataLoader
        kwargs = {'num_workers': value_of(args, "num_workers", 1)}

    original_size = ORIGINAL_SIZE[dataset]
    in_size = args.in_size
    padding = get_padding(original_size, in_size)
    mean, std = get_mean_and_std(dataset, args)

    train_loader = tLoader(
        DATASETS[dataset](f'./.data/.{dataset:s}_data',
                          train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Pad(padding),
                              transforms.ToTensor(),
                              transforms.Lambda(lambda t: t.expand(in_size)),
                              transforms.Normalize((mean,), (std,))
                          ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_dataset = DATASETS[dataset](
        f'./.data/.{dataset:s}_data',
        train=False,
        transform=transforms.Compose([
            transforms.Pad(padding),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.expand(in_size)),
            transforms.Normalize((mean,), (std,))
        ]))

    if args.test_batch_size == 0:
        test_batch_size = len(test_dataset)
    else:
        test_batch_size = args.test_batch_size
    print(f"Test batch size for {dataset:s} will be {test_batch_size:d}.")

    test_loader = tLoader(test_dataset,
                          batch_size=test_batch_size,
                          shuffle=False, **kwargs)

    return train_loader, test_loader


# Create random permutations

def get_permutations(p_no: int, v_size: int, cuda: bool = True) -> LongMatrix:
    fwd_perms = torch.stack([torch.randperm(v_size) for _ in range(p_no)])
    # idxs = torch.linspace(0, v_size - 1, v_size).long()
    if cuda:
        fwd_perms = fwd_perms.cuda()
        # idxs = idxs.cuda()
    # bwd_perms = [p.clone().zero_().index_add_(0, p, idxs) for p in fwd_perms]
    return fwd_perms  # , bwd_perms


def get_full_permutations(dataset: str, p_no: int, args: Args) -> Permutations:
    in_n = reduce(mul, args.in_size, 1)
    i_permutations: LongMatrix = get_permutations(p_no, in_n, args.cuda)
    c_no = CLASSES_NO[dataset]
    o_permutations: Optional[LongMatrix]
    if args.permute_targets:
        o_permutations = get_permutations(p_no, c_no, args.cuda)
    else:
        o_permutations = None
    return (i_permutations, o_permutations)


def get_tasks(args: Args) -> Tasks:
    tasks = {}
    lengths = []
    for task in zip(args.datasets, args.train_batch_size, args.perms_no):
        dataset, batch_size, perms_no = task
        train_loader, test_loader = get_loaders(dataset, batch_size, args)
        perms = get_full_permutations(dataset, perms_no, args)
        tasks[dataset] = DatasetTasks(train_loader, test_loader, perms)
        lengths.append(len(train_loader.dataset))
        print(f"{perms_no:d} tasks for {dataset:s} created.")
    print("Datasets have lengths: ", ", ".join([str(l) for l in lengths]))
    if args.eval_freq == 0:
        if args.mode == "sim":
            args.eval_freq = lengths[0]
        else:
            args.eval_freq = 1
    return tasks

# Apply permutations


def permute(data, target, i_perm: LongVector, t_perm: Optional[LongVector]):
    in_size = data.size()
    data = data.view(in_size[0], -1).index_select(1, i_perm).view(in_size)
    if t_perm is not None:
        target = t_perm.index_select(0, target)
    return data, target


def random_permute(data, target, perms: Permutations):
    i_perms, t_perms = perms
    in_size = data.size()
    in_n = reduce(mul, in_size[1:], 1)
    batch_size, perms_no = data.size(0), i_perms.size(0)
    p_idx = torch.LongTensor(batch_size).random_(0, perms_no)
    if data.is_cuda:
        p_idx = p_idx.cuda()
    idx = i_perms.index_select(0, p_idx).unsqueeze(2)
    data = data.view(batch_size, 1, -1)\
        .expand(batch_size, in_n, in_n)\
        .gather(2, idx)\
        .view(in_size)
    if t_perms is not None:
        target = t_perms.index_select(0, p_idx)\
            .gather(1, target.unsqueeze(1))\
            .squeeze(1)
    return data, target
