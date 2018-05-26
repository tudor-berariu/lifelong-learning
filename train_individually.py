import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from typing import Type, Callable, List, Tuple
from termcolor import colored as clr

# Project imports
from my_types import Args, Tasks, Model, LongVector, DatasetTasks
from multi_task import MultiTask, TaskDataLoader
from utils import AverageMeter, accuracy
from reporting import Accuracy, Loss, show_results, update_results, Reporting


def train(train_loader: TaskDataLoader, model: nn.Module,
          optimizer: torch.optim.Optimizer, epoch: int, report_freq: float = 0.3)-> Tuple[int, int]:

    losses = AverageMeter()
    acc = AverageMeter()
    correct_cnt = 0
    seen = 0

    model.train()

    for batch_idx, (data, target) in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        (top1, correct), = accuracy(output, target)
        correct_cnt += correct

        seen += data.size(0)
        acc.update(top1, data.size(0))
        losses.update(loss.item(), data.size(0))

        if batch_idx % report_freq == 0:
            print(f'\t\t_train_{epoch}_{batch_idx}:\t : Loss: {losses.val:.4f} {losses.avg:.4f}\t'
                  f'Acc: {acc.val:.2f} {acc.avg:.2f}')

    return losses.avg, correct_cnt/seen


def validate(val_loader: TaskDataLoader, model: nn.Module, report_freq: float = 0.1):
    losses = AverageMeter()
    acc = AverageMeter()
    correct_cnt = 0
    seen = 0

    with torch.no_grad():
        for batch_idx, (data, target) in val_loader:
            output = model(data)
            loss = functional.cross_entropy(output, target)
            loss.backward()

            (top1, correct), = accuracy(output, target)
            correct_cnt += correct

            seen += data.size(0)
            acc.update(top1, data.size(0))
            losses.update(loss.item(), data.size(0))

            if batch_idx % report_freq == 0:
                print(
                    f'\t\t_val_{epoch}_{batch_idx}:\t : Loss: {losses.val:.4f} {losses.avg:.4f}\t'
                    f'Acc: {acc.val:.2f} {acc.avg:.2f}')

        return losses.avg, correct_cnt/seen


def train_individually(model_class: Type,
                       get_optimizer: Callable[[nn.Module], optim.Optimizer],
                       multitask: MultiTask,
                       args: Args)-> None:

    print(f"Training {clr('individually', attrs=['bold']):s} on all tasks.")

    epochs_per_task = args.train.epochs_per_task
    model_params = args.model
    batch_report_freq = args.batch_report_freq

    seen, total_epochs_no = 0, 0
    trace, best_results = [], None
    results = {}

    in_size = multitask.in_size
    out_size = multitask.out_size

    train_tasks = multitask.train_tasks()

    report = Reporting(args)

    for task_idx, data_loaders in enumerate(train_tasks):
        train_loader, validate_loader = data_loaders
        dataset_name = train_loader.dataset_name

        print(f"Training on task {task_no:d}: {data_loader.dataset_name:s}.")

        # Initialize model & optim
        model: nn.Module = model_class(model_params, in_size, out_size)
        optimizer = get_optimizer(model.parameters())
        results = {}
        seen = 0
        trace, best_results, not_changed = [], None, 0

        for crt_epoch in range(epochs_per_task):

            # Adjust optimizer learning rate
            # TODO

            train_loss, train_acc = train(train_loader, model, optimizer, crt_epoch,
                                          report_freq=batch_report_freq)
            seen += len(train_loader)
            val_loss, val_acc = validate(validate_loader, model)

            train_info = {"acc": train_loss, "loss": train_acc}
            val_info = {"acc": val_acc, "loss": val_loss}
            results[dataset_name] = val_info
            show_results(seen, results, best_results)

            best_results, changed = update_results(results, best_results)
            not_changed = 0 if changed else (not_changed + 1)
            if not changed:
                print(f"No improvement for {not_changed:d} evals!!")

            report.trace_train(task_idx, crt_epoch, seen, train_info)
            report.trace_eval(task_idx, crt_epoch, seen, val_info)
