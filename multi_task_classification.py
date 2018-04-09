from copy import deepcopy
from typing import Tuple, Dict, List, TypeVar, NewType, NamedTuple, Union
from argparse import Namespace
from functools import reduce
from operator import mul
import os
import os.path
from termcolor import colored as clr

# Torch imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Liftoff : Install https://github.com/tudor-berariu/liftoff

from liftoff.config import read_config, value_of, namespace_to_dict


# Types used in this program

Args = NewType('Args', Namespace)
Loaders = Tuple[DataLoader, DataLoader]
Padding = Tuple[int, int, int, int]
LongVector = TypeVar('LongVector')
MaybeLongVector = Union[LongVector, None]
LongMatrix = TypeVar('LongMatrix')
MaybeLongMatrix = Union[LongMatrix, None]
Permutations = Tuple[LongMatrix, MaybeLongMatrix]
DatasetTasks = NamedTuple("DatasetTasks", [("train_loader", DataLoader),
                                           ("test_loader", DataLoader),
                                           ("perms", Permutations)])
Tasks = Dict[str, DatasetTasks]
Model = NewType('Model', nn.Module)
Accurracy = NewType('Accuracy', float)
Loss = NewType('Loss', float)

EvalResult = NamedTuple(
    "EvalResult",
    [("task_results", Dict[str, Dict[str, List[float]]]),
     ("dataset_avg", Dict[str, Dict[str, float]]),
     ("global_avg", Dict[str, float])]
)
MaybeEvalResult = Union[EvalResult, None]

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


# Read command line arguments

def process_args(args: Args) -> Args:
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if isinstance(args.perms_no, int):
        args.perms_no = [args.perms_no for d in args.datasets]
    elif isinstance(args.perms_no, list) and len(args.perms_no) == 1:
        args.perms_no = args.perms_no * len(args.datasets)
    else:
        assert len(args.perms_no) == len(args.datasets), \
            "You must specify the number of permutations for each dataset."

    args.tasks_no = sum(args.perms_no)

    # Three combinations are supported for now:
    # - a single dataset with any number of permutations
    # - several datasets with the same number of permutations
    # - datasets with different # of permutations, but in this case
    #   the batch size must be a multiple of the total number of tasks

    d_no = len(args.datasets)
    batch_size, perms_no = args.train_batch_size, args.perms_no

    if args.mode != "sim":
        args.train_batch_size = [batch_size] * d_no
    elif d_no == 1:
        args.train_batch_size = [batch_size]
    elif len(set(perms_no)) == 1:
        args.train_batch_size = [batch_size // d_no] * d_no
    else:
        scale = batch_size / sum(perms_no)
        args.train_batch_size = [round(p_no * scale) for p_no in perms_no]

    if args.mode == "sim":
        print(f"Real batch_size will be {sum(args.train_batch_size):d}.")

    sizes = [ORIGINAL_SIZE[d] for d in args.datasets]
    in_sz = torch.Size([max([s[dim] for s in sizes]) for dim in range(3)])
    if args.up_size:
        assert len(args.up_size) == 3, "Please provide a valid volume size."
        in_sz = torch.Size([max(*p) for p in zip(in_sz, args.up_size)])
    args.in_size = in_sz
    print("Input size will be: ", in_sz)

    return args

# Data loaders


def get_padding(in_size: torch.Size, out_size: torch.Size) -> Padding:
    assert len(in_size) == len(out_size)
    d_h, d_w = out_size[-2] - in_size[-2], out_size[-1] - in_size[-1]
    p_h1, p_w1 = d_h // 2, d_w // 2
    p_h2, p_w2 = d_h - p_h1, d_w - p_w1
    return (p_h1, p_h2, p_w1, p_w2)


def get_mean_and_std(dataset: str, args: Args) -> Tuple[float, float]:
    original_size = ORIGINAL_SIZE[dataset]
    in_size = args.in_size
    padding = get_padding(original_size, in_size)
    data = DATASETS[dataset](
        f'./.{dataset:s}_data',
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
    full_data, _ = next(iter(loader))
    mean, std = full_data.mean(), full_data.std()
    del loader, full_data

    return mean, std


def get_loaders(dataset: str, batch_size: int, args: Args) -> Loaders:
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    original_size = ORIGINAL_SIZE[dataset]
    in_size = args.in_size
    padding = get_padding(original_size, in_size)
    mean, std = get_mean_and_std(dataset, args)

    train_loader = DataLoader(
        DATASETS[dataset](f'./.{dataset:s}_data',
                          train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Pad(padding),
                              transforms.ToTensor(),
                              transforms.Lambda(lambda t: t.expand(in_size)),
                              transforms.Normalize((mean,), (std,))
                          ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_dataset = DATASETS[dataset](
        f'./.{dataset:s}_data',
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

    test_loader = DataLoader(test_dataset,
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
    i_permutations = get_permutations(p_no, in_n, args.cuda)
    c_no = CLASSES_NO[dataset]
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


# Models

class MLP(nn.Module):

    def __init__(self, in_size: torch.Size, use_softmax: bool = True):
        super(MLP, self).__init__()
        in_units = reduce(mul, in_size, 1)
        self.fc1 = nn.Linear(in_units, 300)
        self.fc2 = nn.Linear(300, 10)
        self.use_softmax = use_softmax

    def forward(self, *x: List[Variable]) -> Variable:
        x = x[0]
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = self.fc2(F.relu(self.fc1(x)))
        if self.use_softmax:
            return F.softmax(output, dim=-1)
        return output

    def set_softmax(self, use_softmax):
        self.use_softmax = use_softmax


def get_model(args: Args) -> Model:
    return MLP(args.in_size)


def get_optimizer(model: Model, args: Args) -> Optimizer:
    kwargs = value_of(args, "optimizer_args", Namespace(lr=0.001))
    kwargs = namespace_to_dict(kwargs)
    return optim.__dict__[args.optimizer](model.parameters(), **kwargs)


# Apply permutations

def permute(data, target, i_perm: LongVector, t_perm: MaybeLongVector):
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


# Evaluation

def test(model: Model, test_loader: DataLoader,
         i_perm: LongVector, t_perm: MaybeLongVector,
         args: Args) -> Tuple[Accurracy, Loss]:
    model.eval()
    model.set_softmax(use_softmax=False)
    test_loss, correct = 0, 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = permute(data, target, i_perm, t_perm)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target,
                                     size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    model.train()
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    return test_acc, test_loss


def test_all(model: Model, tasks: Tasks, args: Args) -> EvalResult:
    results = {}
    dataset_avg = {}

    for dataset, task in tasks.items():
        accuracies, losses = [], []
        for p_idx in range(task.perms[0].size(0)):
            i_perm = task.perms[0][p_idx]
            t_perm = None if task.perms[1] is None else task.perms[1][p_idx]
            acc, loss = test(model, task.test_loader, i_perm, t_perm, args)
            accuracies.append(acc)
            losses.append(loss)
        results[dataset] = {"acc": accuracies, "loss": losses}
        dataset_avg[dataset] = {
            "acc": np.mean(accuracies),
            "loss": np.mean(losses)
        }

    g_acc = np.concatenate(tuple([r["acc"] for r in results.values()])).mean()
    g_loss = np.concatenate(
        tuple([r["loss"] for r in results.values()])).mean()

    global_avg = {"acc": g_acc, "loss": g_loss}

    return EvalResult(results, dataset_avg, global_avg)


def show_results(idx: int, results: EvalResult, best: MaybeEvalResult) -> None:
    print(''.join("" * 79))
    print(f"Evaluation {clr(f'after {idx:d} examples', attrs=['bold']):s}:")

    for dataset, dataset_results in results.task_results.items():

        accuracies = dataset_results["acc"]
        losses = dataset_results["loss"]
        dataset_best = best.task_results[dataset] if best else None

        for i, (acc, loss) in enumerate(zip(accuracies, losses)):
            msg = f"      " +\
                  f"[Task {clr(f'{dataset:s}-{(i+1):03d}', attrs=['bold']):s}]"

            is_acc_better = dataset_best and dataset_best["acc"][i] < acc
            colors = ['white', 'on_magenta'] if is_acc_better else ['yellow']
            msg += f" Accuracy: {clr(f'{acc:5.2f}%', *colors):s}"

            is_loss_better = dataset_best and dataset_best["loss"][i] > loss
            colors = ['white', 'on_magenta'] if is_loss_better else ['yellow']
            msg += f" Loss: {clr(f'{loss:6.4f}', *colors):s}"

            print(msg)

        msg = f"   [Task {clr(f'{dataset:s}-ALL', attrs=['bold']):s}]"

        d_avg_acc = results.dataset_avg[dataset]["acc"]
        d_avg_loss = results.dataset_avg[dataset]["loss"]

        d_best_avg = best.dataset_avg[dataset] if best else None

        is_acc_better = d_best_avg and d_best_avg["acc"] < d_avg_acc
        colors = ['white', 'on_magenta'] if is_acc_better else ['yellow']
        msg += f" Accuracy: " +\
               f"{clr(f'{d_avg_acc:5.2f}%', *colors, attrs=['bold']):s}"

        is_loss_better = d_best_avg and d_best_avg["loss"] > d_avg_loss
        colors = ['white', 'on_magenta'] if is_loss_better else ['yellow']
        msg += f" Loss: " + \
               f"{clr(f'{d_avg_loss:6.4f}', *colors, attrs=['bold']):s}"

        print(msg)

    msg = f"[{clr('OVERALL', attrs=['bold']):s}]"

    g_acc, g_loss = results.global_avg["acc"], results.global_avg["loss"]

    is_acc_better = best and best.global_avg["acc"] < g_acc
    colors = ['white', 'on_magenta'] if is_acc_better else ['yellow']
    msg += f" Accuracy: {clr(f'{g_acc:5.2f}%', *colors, attrs=['bold']):s}"

    is_loss_better = best and best.global_avg["loss"] > g_loss
    colors = ['white', 'on_magenta'] if is_loss_better else ['yellow']
    msg += f" Loss: {clr(f'{g_loss:6.4f}', *colors, attrs=['bold']):s}"

    print(msg)


def update_results(results: EvalResult,
                   best: MaybeEvalResult) -> Tuple[EvalResult, bool]:
    from operator import lt, gt

    if not best:
        return deepcopy(results), True
    changed = False

    for dataset, dataset_results in results.task_results.items():
        best_results = best.task_results[dataset]

        acc_pairs = zip(dataset_results["acc"], best_results["acc"])
        changed = changed or any(map(lambda p: gt(*p), acc_pairs))
        acc_pairs = zip(dataset_results["acc"], best_results["acc"])
        best_results["acc"] = [max(*p) for p in acc_pairs]

        loss_pairs = zip(dataset_results["loss"], best_results["loss"])
        changed = changed or any(map(lambda p: lt(*p), acc_pairs))
        loss_pairs = zip(dataset_results["loss"], best_results["loss"])
        best_results["loss"] = [min(*p) for p in loss_pairs]

        crt_avg = results.dataset_avg[dataset]
        best_avg = best.dataset_avg[dataset]

        changed = changed or gt(crt_avg["acc"], best_avg["acc"])
        best_avg["acc"] = max(crt_avg["acc"], best_avg["acc"])
        changed = changed or lt(crt_avg["loss"], best_avg["loss"])
        best_avg["loss"] = min(crt_avg["loss"], best_avg["loss"])

    crt_g_avg = results.global_avg
    best_g_avg = best.global_avg

    changed = changed or gt(crt_g_avg["acc"], best_g_avg["acc"])
    best_g_avg["acc"] = max(crt_g_avg["acc"], best_g_avg["acc"])
    changed = changed or lt(crt_g_avg["loss"], best_g_avg["loss"])
    best_g_avg["loss"] = min(crt_g_avg["loss"], best_g_avg["loss"])

    return best, changed


# Training procedures

def train_simultaneously(model: nn.Module,
                         optimizer: optim.Optimizer,
                         tasks: Tasks,
                         args: Args)-> None:
    print(f"Training {clr('simultaneously', attrs=['bold']):s} on all tasks.")

    task_lps = [(len(task.train_loader.dataset), task.perms[0].size(0))
                for task in tasks.values()]
    all_tasks_no = sum([p for (_, p) in task_lps])
    avg_length = sum([l * p for (l, p) in task_lps]) / all_tasks_no
    max_inputs = int(avg_length * args.epochs_per_task * all_tasks_no)
    eval_freq = int(avg_length * args.eval_freq)

    print(f"Model will be trained for a maximum of {max_inputs:d} samples.")
    print(f"Model will be evaluated every {eval_freq:d} training samples.")

    model.train()
    model.set_softmax(use_softmax=False)

    train_iterators = {d: iter(dts.train_loader) for (d, dts) in tasks.items()}
    seen, next_eval = 0, eval_freq
    trace, best_results, not_changed = [], None, 0

    while seen < max_inputs:
        full_data, full_target = [], []
        for dataset in args.datasets:
            train_iterator = train_iterators[dataset]
            try:
                (data, target) = next(train_iterator)
            except StopIteration:
                train_iterator = iter(tasks[dataset].train_loader)
                train_iterators[dataset] = train_iterator
                (data, target) = next(train_iterator)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = random_permute(data, target, tasks[dataset].perms)
            full_data.append(data)
            full_target.append(target)

        if len(full_data) > 1:
            full_data = torch.cat(full_data, dim=0)
            full_target = torch.cat(full_target, dim=0)
        else:
            full_data, full_target = full_data[0], full_target[0]

        optimizer.zero_grad()
        output = model(Variable(full_data))
        loss = F.cross_entropy(output, Variable(full_target))
        loss.backward()
        optimizer.step()

        seen += full_data.size(0)

        if seen >= next_eval:
            results = test_all(model, tasks, args)
            model.train()
            model.set_softmax(use_softmax=False)
            show_results(seen, results, best_results)
            best_results, changed = update_results(results, best_results)
            next_eval += eval_freq
            not_changed = 0 if changed else (not_changed + 1)
            if not changed:
                print(f"No improvement for {not_changed:d} evals!!")
            trace.append((seen, -1, results))
            if len(trace) % args.save_freq == 0:
                torch.save(trace, os.path.join(args.out_dir,
                                               f"seen_{seen:d}_trace.pkl"))
            if not_changed > 0 and args.stop_if_not_better == not_changed:
                break

    if seen != trace[-1][0]:
        results = test_all(model, tasks, args)
        show_results(seen, results, best_results)
        trace.append((seen, -1, results))

    torch.save(trace, os.path.join(args.out_dir, f"trace.pkl"))
    return trace


def order_tasks(task_args: List[Tuple[str, int]], args: Args) -> None:
    """In place ordering of tasks"""

    if args.tasks_order == "shuffle_all":
        np.random.shuffle(task_args)
    elif isinstance(args.tasks_order, list) and args.tasks_order[0] == "alt":
        task_args.sort(key=lambda p: args.tasks_order.index(p[0]) + p[1] * 100)
    elif isinstance(args.tasks_order, list):
        task_args.sort(key=lambda p: args.tasks_order.index(p[0]))
    else:
        raise NotImplementedError


def train_in_sequence(model: nn.Module,
                      optimizer: optim.Optimizer,
                      tasks: Tasks,
                      args: Args)-> None:
    print(f"Training all tasks {clr('in sequence', attrs=['bold']):s}.")
    if args.mode == "ewc":
        print("Using elastic weight consolidation.")
    elif args.mode == "apc":
        print("Using action preserving constraints")

    model.train()
    model.set_softmax(use_softmax=False)

    seen, total_epochs_no = 0, 0
    trace, best_results = [], None
    task_args = [(d_name, p_idx)
                 for d_name, p_no in zip(args.datasets, args.perms_no)
                 for p_idx in range(p_no)]

    order_tasks(task_args, args)

    print("Tasks order will be: ",
          ", ".join([f"{dataset:s}-{(p_idx+1):03d}"
                     for (dataset, p_idx) in task_args]))

    for task_no, (dataset, p_idx) in enumerate(task_args):
        print(f"Training on task {task_no:d}: {dataset:s}-{(p_idx+1):03d}.")
        not_changed = 0
        crt_epochs = 0
        task = tasks[dataset]
        i_perm = task.perms[0][p_idx]
        t_perm = None if task.perms[1] is None else task.perms[1][p_idx]

        while crt_epochs < args.epochs_per_task:

            for data, target in task.train_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = permute(data, target, i_perm, t_perm)

                optimizer.zero_grad()
                output = model(Variable(data))
                loss = F.cross_entropy(output, Variable(target))
                loss.backward()
                optimizer.step()

            seen += len(task.train_loader.dataset)
            crt_epochs += 1
            total_epochs_no += 1

            if total_epochs_no % args.eval_freq == 0:
                results = test_all(model, tasks, args)
                model.train()
                model.set_softmax(use_softmax=False)
                show_results(seen, results, best_results)
                best_results, changed = update_results(results, best_results)
                not_changed = 0 if changed else (not_changed + 1)
                if not changed:
                    print(f"No improvement for {not_changed:d} evals!!")
                trace.append((seen, total_epochs_no, results))
                if len(trace) % args.save_freq == 0:
                    torch.save(trace,
                               os.path.join(args.out_dir,
                                            f"epoch_{total_epochs_no:d}_trace.pkl"))
                if not_changed > 0 and args.stop_if_not_better == not_changed:
                    break

    torch.save(trace, os.path.join(args.out_dir, f"trace.pkl"))
    return trace


def run(args: Args) -> None:
    args = process_args(args)

    # Model, optimizer, tasks
    model = get_model(args)
    if args.cuda:
        model.cuda()
    optimizer = get_optimizer(model, args)
    tasks = get_tasks(args)

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    if args.mode == "sim":
        train_simultaneously(model, optimizer, tasks, args)
    else:
        train_in_sequence(model, optimizer, tasks, args)


def main():

    # Reading args
    args = read_config()

    import time
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    out_dir = f'./results/{str(int(time.time())):s}_{args.experiment:s}'
    args.out_dir = out_dir
    args.run_id = 0

    run(args)


if __name__ == "__main__":
    main()
