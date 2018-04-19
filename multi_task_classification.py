from copy import deepcopy
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
import os
import os.path
from termcolor import colored as clr
#  from tqdm import tqdm
import pandas as pd

# Torch imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Liftoff : Install https://github.com/tudor-berariu/liftoff

from liftoff.config import read_config

# Project imports
from models import get_model, get_optimizer
from tasks import ORIGINAL_SIZE, get_tasks, permute, random_permute
from elastic_constraints import elastic_loss

# Types across modules

from my_types import Args, Tasks, Model, LongVector

# Local types

Accuracy = float
Loss = float

EvalResult = NamedTuple(
    "EvalResult",
    [("task_results", Dict[str, Dict[str, List[float]]]),
     ("dataset_avg", Dict[str, Dict[str, float]]),
     ("global_avg", Dict[str, float])]
)


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

# Evaluation


def test(model: Model, test_loader: DataLoader,
         i_perm: LongVector, t_perm: Optional[LongVector],
         args: Args) -> Tuple[Accuracy, Loss]:
    model.eval()
    model.use_softmax = False
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


def show_results(seen: int, results: EvalResult,
                 best: Optional[EvalResult]) -> None:
    print(''.join("" * 79))
    print(f"Evaluation {clr(f'after {seen:d} examples', attrs=['bold']):s}:")

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
                   best: Optional[EvalResult]) -> Tuple[EvalResult, bool]:
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


def results_to_dataframe(results: EvalResult) -> pd.DataFrame:
    return pd.concat([pd.DataFrame({
        'task': [f"{dataset:s}-{(i+1):d}" for i in range(len(d_results["acc"]))],
        'acc': d_results["acc"],
        'loss': d_results["loss"]
    }) for (dataset, d_results) in results.task_results.items()])\
        .reset_index(drop=True)


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
    model.use_softmax = False

    train_iterators = {d: iter(dts.train_loader) for (d, dts) in tasks.items()}
    seen, next_eval = 0, eval_freq
    trace, best_results, not_changed = [], None, 0

    while seen < max_inputs:
        inputs: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []

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
            inputs.append(data)
            targets.append(target)

        full_data: torch.Tensor
        full_target: torch.Tensor
        if len(inputs) > 1:
            full_data = torch.cat(inputs, dim=0)
            full_target = torch.cat(targets, dim=0)
        else:
            full_data, full_target = inputs[0], targets[0]

        optimizer.zero_grad()
        output = model(Variable(full_data))
        loss = F.cross_entropy(output, Variable(full_target))
        loss.backward()
        optimizer.step()

        seen += full_data.size(0)

        if seen >= next_eval:
            results = test_all(model, tasks, args)
            model.train()
            model.use_softmax = False
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


def add_exp_params(dfr: pd.DataFrame, args: Args) -> None:
    if hasattr(args, "_experiment_parameters"):
        for p_name, p_value in args._experiment_parameters.__dict__.items():
            if isinstance(p_value, list):
                dfr[p_name] = ",".join([str(el) for el in p_value])
            elif isinstance(p_value, dict):
                dfr[p_name] = ",".join([str(key) + "=" + str(el)
                                        for (key, el) in p_value.items()])
            else:
                dfr[p_name] = p_value
    if hasattr(args, "title"):
        dfr['title'] = args.title
    if hasattr(args, "run_id"):
        dfr['run_id'] = args.run_id


def train_in_sequence(model: nn.Module,
                      optimizer: optim.Optimizer,
                      tasks: Tasks,
                      args: Args) -> None:
    print(f"Training all tasks {clr('in sequence', attrs=['bold']):s}.")
    if args.mode == "elastic":
        print(f"... using {clr('elastic constraints', attrs=['bold']):s}.")

    model.train()
    model.use_softmax = False

    seen, total_epochs_no = 0, 0
    trace, best_results = [], None
    task_args = [(d_name, p_idx)
                 for d_name, p_no in zip(args.datasets, args.perms_no)
                 for p_idx in range(p_no)]

    order_tasks(task_args, args)

    print("Tasks order will be: ",
          ", ".join([f"{dataset:s}-{(p_idx+1):03d}"
                     for (dataset, p_idx) in task_args]))
    elastic: Optional[Callable[[nn.Module], Variable]] = None
    eval_df = None
    train_info: Dict[str, List[Any]] = {
        'seen': [], 'epoch': [], 'ce_loss': [], 'ec_loss': []}

    for task_no, (dataset, p_idx) in enumerate(task_args):
        print(f"Training on task {task_no:d}: {dataset:s}-{(p_idx+1):03d}.")
        not_changed = 0
        crt_epochs = 0
        task = tasks[dataset]
        i_perm = task.perms[0][p_idx]
        t_perm = None if task.perms[1] is None else task.perms[1][p_idx]

        if task_no > 0:
            elastic = elastic_loss(model, tasks, task_args[:task_no], args)

        while crt_epochs < args.epochs_per_task:

            ce_losses = []
            ec_losses = []

            for data, target in task.train_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = permute(data, target, i_perm, t_perm)

                optimizer.zero_grad()
                output = model(Variable(data))
                loss = F.cross_entropy(output, Variable(target))
                ce_losses.append(loss.data[0])
                if elastic is not None:
                    elastic_penalty = args.elasticity.scale * elastic(model)
                    ec_losses.append(elastic_penalty.data[0])
                    if elastic_penalty.data[0] > 0:
                        loss += elastic_penalty

                loss.backward()
                optimizer.step()

            seen += len(task.train_loader.dataset)
            crt_epochs += 1
            total_epochs_no += 1

            train_info['seen'].append(seen)
            train_info['epoch'].append(total_epochs_no)
            train_info['ce_loss'].append(np.mean(ce_losses))
            train_info['ec_loss'].append(
                np.mean(ec_losses) if ec_losses else 0)

            if total_epochs_no % args.eval_freq == 0:
                results = test_all(model, tasks, args)
                model.train()
                model.use_softmax = False
                show_results(seen, results, best_results)
                best_results, changed = update_results(results, best_results)
                not_changed = 0 if changed else (not_changed + 1)
                if not changed:
                    print(f"No improvement for {not_changed:d} evals!!")
                trace.append((seen, total_epochs_no, results))
                e_df = results_to_dataframe(results)
                e_df['seen'] = seen
                e_df['epoch'] = total_epochs_no
                add_exp_params(e_df, args)
                if eval_df is None:
                    eval_df = e_df
                else:
                    eval_df = pd.concat([eval_df, e_df]).reset_index(drop=True)

                if len(trace) % args.save_freq == 0:
                    train_df = pd.DataFrame(train_info)
                    add_exp_params(train_df, args)

                    train_df.to_pickle(os.path.join(
                        args.out_dir,
                        f"epoch_{total_epochs_no:d}_losses.pkl"))
                    eval_df.to_pickle(os.path.join(
                        args.out_dir,
                        f"epoch_{total_epochs_no:d}_results.pkl"))
                    torch.save(trace, os.path.join(
                        args.out_dir,
                        f"epoch_{total_epochs_no:d}_trace.pkl"))
                if not_changed > 0 and args.stop_if_not_better == not_changed:
                    break

    train_df = pd.DataFrame(train_info)
    train_df['title'] = args.title
    add_exp_params(train_df, args)
    train_df.to_pickle(os.path.join(args.out_dir, f"losses.pkl"))
    eval_df.to_pickle(os.path.join(args.out_dir, f"results.pkl"))
    torch.save(trace, os.path.join(args.out_dir, f"trace.pkl"))


def run(args: Args) -> None:
    args = process_args(args)

    # Model, optimizer, tasks
    model = get_model(args)  # type: Model
    if args.cuda:
        model.cuda()
    optimizer = get_optimizer(model, args)  # type: optim.Optimizer
    tasks = get_tasks(args)  # type: Tasks

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    if args.mode == "sim":
        train_simultaneously(model, optimizer, tasks, args)
    else:
        train_in_sequence(model, optimizer, tasks, args)


def main():

    # Reading args
    args = read_config()  # type: Args

    if not hasattr(args, "out_dir"):
        from time import time
        if not os.path.isdir('./results'):
            os.mkdir('./results')
        out_dir = f'./results/{str(int(time())):s}_{args.experiment:s}'
        os.mkdir(out_dir)
        args.out_dir = out_dir
    else:
        assert os.path.isdir(args.out_dir), "Given directory does not exist"

    if not hasattr(args, "run_id"):
        args.run_id = 0

    run(args)


if __name__ == "__main__":
    main()
