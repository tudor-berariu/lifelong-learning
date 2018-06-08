import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from typing import Callable, Any
from termcolor import colored as clr

# Project imports
from my_types import Args
from multi_task import MultiTask
from utils.util import standard_train, standard_validate
from utils.reporting import Reporting


def train_individually(init_model: Callable[[Any], nn.Module],
                       get_optimizer: Callable[[nn.Module], optim.Optimizer],
                       multitask: MultiTask,
                       args: Args)-> None:

    print(f"Training {clr('individually', attrs=['bold']):s} on all tasks.")

    epochs_per_task = args.train.epochs_per_task
    model_params = args.model

    # TODO: check this [0][0] below
    # You said report means saving, but it seems to print...
    batch_train_show_freq = args.reporting.batch_train_show_freq

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

            report.trace_train(seen, task_idx, crt_epoch, train_info)
            new_best_acc, new_best_loss = report.trace_eval(seen, task_idx, crt_epoch,
                                                            val_epoch, val_info)

            if crt_epoch % save_report_freq == 0:
                report.save()

        report.finished_training_task(task_idx+1, seen)

    report.save(final=True)
