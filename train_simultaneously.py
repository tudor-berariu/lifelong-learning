import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from typing import Type, Callable
from termcolor import colored as clr
import numpy as np

# Project imports
from my_types import Args
from multi_task import MultiTask
from utils.util import standard_train, standard_validate
from utils.reporting import Reporting


def train_simultaneously(model_class: Type,
                         get_optimizer: Callable[[nn.Module], optim.Optimizer],
                         multitask: MultiTask,
                         args: Args)-> None:

    print(f"Training {clr('simultaneously', attrs=['bold']):s} on all tasks.")

    epochs_per_task = args.train.epochs_per_task
    stop_if_not_better = args.train.stop_if_not_better
    max_nan_loss = args.train.max_nan_loss
    model_params = args.model
    batch_train_show_freq = args.reporting.batch_train_show_freq

    in_size = multitask.in_size
    out_size = multitask.out_size

    no_tasks = len(multitask)
    train_loader = multitask.merged_tasks()
    train_batch_cnt = int(multitask.average_batches_per_epoch / no_tasks)
    train_task_idx = 0

    # Initialize model & optim
    model: nn.Module = model_class(model_params, in_size, out_size)
    optimizer = get_optimizer(model.parameters())

    report = Reporting(args, multitask.get_task_info(), model_summary=model)
    save_report_freq = args.reporting.save_report_freq

    seen = 0
    val_epoch = 0
    not_better = 0
    no_nan_loss = 0

    # -- LR Scheduler
    optim_args = args.train._optimizer
    if hasattr(optim_args, "lr_decay"):
        step = optim_args.lr_decay.step
        gamma = optim_args.lr_decay.gamma
        scheduler = MultiStepLR(optimizer,
                                milestones=list(range(step * no_tasks, epochs_per_task * no_tasks,
                                                      step * no_tasks)), gamma=gamma)
    else:
        scheduler = None

    for crt_epoch in range(epochs_per_task * no_tasks):
        # TODO Adjust optimizer learning rate
        if scheduler:
            scheduler.step()
        train_loss, train_acc, train_seen = standard_train(train_loader, model, optimizer,
                                                           crt_epoch,
                                                           batch_show_freq=batch_train_show_freq,
                                                           max_batch=train_batch_cnt)
        seen += train_seen

        train_info = {"acc": train_acc, "loss": train_loss}
        report.trace_train(seen, train_task_idx, crt_epoch, train_info)

        all_tasks_new_best_acc = 0
        all_tasks_new_best_loss = 0
        for task_idx, validate_loader in enumerate(multitask.test_tasks(no_tasks)):
            val_loss, val_acc = standard_validate(
                validate_loader, model, crt_epoch)

            #  -- Reporting
            val_info = {"acc": val_acc, "loss": val_loss}

            new_best_acc, new_best_loss = report.trace_eval(seen, task_idx, crt_epoch,
                                                            val_epoch, val_info)
            all_tasks_new_best_acc += new_best_acc
            all_tasks_new_best_loss += new_best_loss

        # Check improvements
        if all_tasks_new_best_acc + all_tasks_new_best_loss > 0:
            not_better = 0
        else:
            not_better += 1
            if not_better > stop_if_not_better:
                print(f"Stop training because of {not_better} epochs without improvement.")
                break

        # Check NaN loss
        if np.isnan(train_loss):
            no_nan_loss += 1
            if no_nan_loss > max_nan_loss:
                print(f"Stop training because of {no_nan_loss} epochs with loss NaN.")
                break
        else:
            no_nan_loss = 0

        val_epoch += 1

        if crt_epoch % save_report_freq == 0:
            report.save()

        report.finished_training_task(no_tasks, seen)

    report.save(final=True)
