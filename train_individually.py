import torch
import torch.nn as nn
import torch.optim as optim
from typing import Type, Callable, Tuple, Any
from termcolor import colored as clr

# Project imports
from my_types import Args
from multi_task import MultiTask
from utils import standard_train, standard_validate
from reporting import Reporting


def train_individually(model_class: Callable[[Any], nn.Module],
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

        report.register_model(model)

        seen = 0
        val_epoch = 0

        for crt_epoch in range(epochs_per_task):

            # TODO Adjust optimizer learning rate

            train_loss, train_acc, _ = standard_train(train_loader, model, optimizer, crt_epoch,
                                                      report_freq=batch_report_freq)
            seen += len(train_loader)

            val_loss, val_acc = standard_validate(validate_loader, model, crt_epoch)
            val_epoch += 1

            #  -- Reporting
            train_info = {"acc": train_acc, "loss": train_loss}
            val_info = {"acc": val_acc, "loss": val_loss}

            report.trace_train(seen, task_idx, crt_epoch, train_info)
            new_best_acc, new_best_loss = report.trace_eval(seen, task_idx, crt_epoch,
                                                            val_epoch, val_info)

            if crt_epoch % save_report_freq == 0:
                report.save()

        report.finished_training_task(task_idx+1, seen)

    report.save()
