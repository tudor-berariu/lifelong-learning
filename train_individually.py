import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from typing import Type, Callable, Tuple
from termcolor import colored as clr

# Project imports
from my_types import Args
from multi_task import MultiTask, TaskDataLoader
from utils import AverageMeter, accuracy
from reporting import Reporting


def train(train_loader: TaskDataLoader, model: nn.Module,
          optimizer: torch.optim.Optimizer, epoch: int, report_freq: float = 0.3)-> Tuple[int, int]:

    losses = AverageMeter()
    acc = AverageMeter()
    correct_cnt = 0
    seen = 0.

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
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

        if (batch_idx + 1) % report_freq == 0:
            print(f'\t\t[Train] [Epoch: {epoch:3}] [Batch: {batch_idx:5}]:\t '
                  f'[Loss] crt: {losses.val:3.4f}  avg: {losses.avg:3.4f}\t'
                  f'[Accuracy] crt: {acc.val:3.2f}  avg: {acc.avg:.2f}')

    return losses.avg, correct_cnt / float(seen)


def validate(val_loader: TaskDataLoader, model: nn.Module, epoch: int, report_freq: float = 0.1):
    losses = AverageMeter()
    acc = AverageMeter()
    correct_cnt = 0
    seen = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            output = model(data)
            loss = functional.cross_entropy(output, target)

            (top1, correct), = accuracy(output, target)
            correct_cnt += correct

            seen += data.size(0)
            acc.update(top1, data.size(0))
            losses.update(loss.item(), data.size(0))

            if (batch_idx + 1) % report_freq == 0:
                print(
                    f'\t\t_val_{epoch}_{batch_idx}:\t : Loss: {losses.val:.4f} {losses.avg:.4f}\t'
                    f'Acc: {acc.val:.2f} {acc.avg:.2f}')

        return losses.avg, correct_cnt / float(seen)


def train_individually(model_class: Type,
                       get_optimizer: Callable[[nn.Module], optim.Optimizer],
                       multitask: MultiTask,
                       args: Args)-> None:

    print(f"Training {clr('individually', attrs=['bold']):s} on all tasks.")

    epochs_per_task = args.train.epochs_per_task
    model_params = args.model
    batch_report_freq = args.reporting.batch_report_freq

    in_size = multitask.in_size
    out_size = multitask.out_size

    train_tasks = multitask.train_tasks()

    report = Reporting(args, multitask.get_task_info())
    save_report_freq = args.reporting.save_report_freq

    for task_idx, data_loaders in enumerate(train_tasks):
        train_loader, validate_loader = data_loaders
        task_name = train_loader.name

        print(f"Training on task {task_idx:d}: {task_name:s}.")

        # Initialize model & optim
        model: nn.Module = model_class(model_params, in_size, out_size)
        optimizer = get_optimizer(model.parameters())

        seen = 0
        val_epoch = 0

        for crt_epoch in range(epochs_per_task):

            # TODO Adjust optimizer learning rate

            train_loss, train_acc = train(train_loader, model, optimizer, crt_epoch,
                                          report_freq=batch_report_freq)
            seen += len(train_loader)

            val_loss, val_acc = validate(validate_loader, model, crt_epoch)
            val_epoch += 1

            #  -- Reporting
            train_info = {"acc": train_acc, "loss": train_loss}
            val_info = {"acc": val_acc, "loss": val_loss}

            report.trace_train(seen, task_idx, crt_epoch, train_info)
            new_best_acc, new_best_loss = report.trace_eval(seen, task_idx, crt_epoch,
                                                            val_epoch, val_info)

            if crt_epoch % save_report_freq == 0:
                report.save()

    report.save()

