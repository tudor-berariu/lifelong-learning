import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from typing import Callable, Any, Tuple, Union, Iterator
from termcolor import colored as clr
import torch
import torch.nn.functional as functional
import numpy as np
import copy

# Project imports
from my_types import Args
from multi_task import MultiTask, TaskDataLoader, Batch
from utils.util import AverageMeter, accuracy
from utils.reporting import Reporting

LOW = 10
HIGH = 15


def standard_train(train_loader: Union[TaskDataLoader, Iterator[Batch]], model: nn.Module,
                   optimizer: torch.optim.Optimizer, epoch: int,
                   batch_show_freq: int = -1,
                   max_batch: int = np.inf)-> Tuple[float, float, int]:

    losses = AverageMeter()
    acc = AverageMeter()
    correct_cnt = 0
    seen = 0.

    model.train()

    for batch_idx, (data, targets, head_idx) in enumerate(train_loader):
        if batch_idx > max_batch:
            break

        optimizer.zero_grad()
        outputs = model(data, head_idx=head_idx)

        loss = 0
        for out, target in zip(outputs, targets):
            new_target: torch.Tensor = torch.zeros_like(out)
            new_target.scatter_(1, target.unsqueeze(1), torch.zeros_like(target).random_(LOW,
                                                                                         HIGH).float().unsqueeze(1))
            loss += functional.smooth_l1_loss(out, new_target)

        loss.backward()
        optimizer.step()

        (top1, correct), = accuracy(outputs, targets)
        correct_cnt += correct

        seen += data.size(0)
        acc.update(top1, data.size(0))
        losses.update(loss.item(), data.size(0))

        if batch_show_freq > 0 and (batch_idx + 1) % batch_show_freq == 0:
            print(f'\t\t[Train] [Epoch: {epoch:3}] [Batch: {batch_idx:5}]:\t '
                  f'[Loss] crt: {losses.val:3.4f}  avg: {losses.avg:3.4f}\t'
                  f'[Accuracy] crt: {acc.val:3.2f}  avg: {acc.avg:.2f}')

    return losses.avg, correct_cnt / float(seen), seen


def standard_validate(val_loader: TaskDataLoader, model: nn.Module, epoch: int,
                      report_freq: int = -1):
    losses = AverageMeter()
    acc = AverageMeter()
    correct_cnt = 0
    seen = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (data, targets, head_idx) in enumerate(val_loader):
            outputs = model(data, head_idx=head_idx)

            loss = 0
            for out, target in zip(outputs, targets):
                new_target: torch.Tensor = torch.zeros_like(out)
                new_target.scatter_(1, target.unsqueeze(1), np.random.randint(4, 7))
                loss += functional.smooth_l1_loss(out, new_target)
            print(out[0, target[0]])

            (top1, correct), = accuracy(outputs, targets)
            correct_cnt += correct

            seen += data.size(0)
            acc.update(top1, data.size(0))
            losses.update(loss.item(), data.size(0))

            if (batch_idx + 1) % report_freq == 0 and report_freq > 0:
                print(f'\t\t[Eval] [Epoch: {epoch:3}] [Batch: {batch_idx:5}]:\t '
                      f'[Loss] crt: {losses.val:3.4f}  avg: {losses.avg:3.4f}\t'
                      f'[Accuracy] crt: {acc.val:3.2f}  avg: {acc.avg:.2f}')

        return losses.avg, correct_cnt / float(seen)


def test_variance(init_model: Callable[[Any], nn.Module],
                  get_optimizer: Callable[[nn.Module], optim.Optimizer],
                  multitask: MultiTask,
                  args: Args)-> None:

    print(f"Training {clr('with random random variance on label', attrs=['bold']):s} on all tasks.")

    epochs_per_task = args.train.epochs_per_task
    model_params = args.model

    # TODO: check this [0][0] below
    # You said report means saving, but it seems to print...
    batch_train_show_freq = args.reporting.batch_train_show_freq

    in_size = multitask.in_size
    out_size = multitask.out_size

    train_tasks = multitask.train_tasks()

    rep_args = copy.deepcopy(args)
    rep_args.mode = "ind"  # Use same reporting

    report = Reporting(rep_args, multitask.get_task_info())
    save_report_freq = args.reporting.save_report_freq
    all_epoch = 0
    all_val_epoch = 0

    for task_idx, data_loaders in enumerate(train_tasks):
        train_loader, validate_loader = data_loaders
        task_name = train_loader.name

        print(f"Training on task {task_idx:d}: {task_name:s}.")

        # Initialize model & optim
        model: nn.Module = init_model(model_params, in_size, out_size)
        optimizer = get_optimizer(model.parameters())

        report.register_model({"summary": model.__str__()})

        # -- LR Scheduler
        optim_args = args.train._optimizer
        if hasattr(optim_args, "lr_decay"):
            step = optim_args.lr_decay.step
            gamma = optim_args.lr_decay.gamma
            scheduler = MultiStepLR(optimizer, milestones=list(range(step, epochs_per_task, step)),
                                    gamma=gamma)
        else:
            scheduler = None

        seen = 0
        val_epoch = 0

        for crt_epoch in range(epochs_per_task):
            if scheduler:
                scheduler.step()
            # TODO Adjust optimizer learning rate

            train_loss, train_acc, _ = standard_train(train_loader, model, optimizer, crt_epoch,
                                                      batch_show_freq=batch_train_show_freq)
            seen += len(train_loader)

            val_loss, val_acc = standard_validate(validate_loader, model, crt_epoch)
            val_epoch += 1

            #  -- Reporting
            train_info = {"acc": train_acc, "loss": train_loss}
            val_info = {"acc": val_acc, "loss": val_loss}

            report.trace_train(seen, task_idx, crt_epoch, all_epoch, train_info)
            new_best_acc, new_best_loss = report.trace_eval(seen, task_idx, crt_epoch,
                                                            val_epoch, all_epoch, val_info)

            if crt_epoch % save_report_freq == 0:
                report.save()

            all_epoch += 1

        report.finished_training_task(task_idx+1, seen)

    report.save(final=True)
