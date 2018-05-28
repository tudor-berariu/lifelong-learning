import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from typing import Type, Callable, Tuple, Iterator
from termcolor import colored as clr

# Project imports
from my_types import Args, Tasks, Model, LongVector, DatasetTasks
from multi_task import MultiTask


# Project imports
from my_types import Args
from multi_task import MultiTask, TaskDataLoader, Batch
from utils import AverageMeter, accuracy
from reporting import Reporting


def train(train_loader: Iterator[Batch], max_batch: int, model: nn.Module,
          optimizer: torch.optim.Optimizer,
          epoch: int, report_freq: float = 0.3)-> Tuple[int, int, int]:

    losses = AverageMeter()
    acc = AverageMeter()
    correct_cnt = 0
    seen = 0.

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= max_batch:
            break

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

    return losses.avg, correct_cnt / float(seen), seen


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


def train_simultaneously(model_class: Type,
                         get_optimizer: Callable[[nn.Module], optim.Optimizer],
                         multitask: MultiTask,
                         args: Args)-> None:

    print(f"Training {clr('simultaneously', attrs=['bold']):s} on all tasks.")

    epochs_per_task = args.train.epochs_per_task
    model_params = args.model
    batch_report_freq = args.reporting.batch_report_freq

    in_size = multitask.in_size
    out_size = multitask.out_size

    train_loader = multitask.merged_tasks()
    train_batch_cnt = multitask.get_merged_tasks_estimated_batches_cnt()

    report = Reporting(args, multitask.get_task_info())
    save_report_freq = args.reporting.save_report_freq

    # Initialize model & optim
    model: nn.Module = model_class(model_params, in_size, out_size)
    optimizer = get_optimizer(model.parameters())

    seen = 0
    val_epoch = 0
    no_tasks = len(multitask)

    for crt_epoch in range(epochs_per_task):
        # TODO Adjust optimizer learning rate

        train_loss, train_acc, train_seen = train(train_loader, train_batch_cnt, model, optimizer,
                                                  crt_epoch, report_freq=batch_report_freq)
        seen += train_seen

        for task_idx, validate_loader in enumerate(multitask.test_tasks(no_tasks)):
            val_loss, val_acc = validate(validate_loader, model, crt_epoch)

            #  -- Reporting
            train_info = {"acc": train_acc, "loss": train_loss}
            val_info = {"acc": val_acc, "loss": val_loss}

            print(val_loss)
            report.trace_train(seen, task_idx, crt_epoch, train_info)
            new_best_acc, new_best_loss = report.trace_eval(seen, task_idx, crt_epoch,
                                                            val_epoch, val_info)

        val_epoch += 1

        if crt_epoch % save_report_freq == 0:
            report.save()

    report.save()
