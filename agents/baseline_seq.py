import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from typing import Any, Callable
import os

# Project imports
from my_types import Args
from multi_task import MultiTask
from utils import standard_train, standard_validate
from utils.reporting import Reporting


def train_sequentially(model_class: Callable[[Any], nn.Module],
                       get_optimizer: Callable[[nn.Module], optim.Optimizer],
                       multitask: MultiTask,
                       args: Args)-> None:

    epochs_per_task = args.train.epochs_per_task
    model_params = args.model
    batch_report_freq = args.reporting.batch_report_freq
    eval_freq = args.reporting.eval_freq
    eval_not_trained = args.reporting.eval_not_trained

    in_size = multitask.in_size
    out_size = multitask.out_size

    # Initialize model & optim
    model: nn.Module = model_class(model_params, in_size, out_size)
    optimizer = get_optimizer(model.parameters())

    train_tasks = multitask.train_tasks()

    report = Reporting(args, multitask.get_task_info(), model_summary=model,
                       files_to_save=[os.path.abspath(__file__)])

    save_report_freq = args.reporting.save_report_freq
    seen = 0
    no_tasks = len(multitask)
    all_epochs = 0
    all_val_epochs = 0

    # -- LR Scheduler

    if hasattr(args.train, "_lr_decay"):
        step = args.train._lr_decay.step
        gamma = args.train._lr_decay.gamma
        scheduler = MultiStepLR(optimizer,
                                milestones=list(range(step,
                                                      epochs_per_task * no_tasks,
                                                      step)),
                                gamma=gamma)
    else:
        scheduler = None

    for train_task_idx, data_loaders in enumerate(train_tasks):
        train_loader, validate_loader = data_loaders
        task_name = train_loader.name

        print(f"Training on task {train_task_idx:d}: {task_name:s}.")

        val_epoch = 0

        for crt_epoch in range(epochs_per_task):
            if scheduler:
                scheduler.step()

            train_loss, train_acc, _ = standard_train(train_loader, model, optimizer, crt_epoch,
                                                      report_freq=batch_report_freq)
            seen += len(train_loader)

            train_info = {"acc": train_acc, "loss": train_loss}
            report.trace_train(seen, train_task_idx, crt_epoch, all_epochs, train_info)

            # Evaluate
            if crt_epoch % eval_freq == 0 or crt_epoch == (epochs_per_task - 1):
                how_many = no_tasks if eval_not_trained else train_task_idx + 1
                for test_task_idx, validate_loader in enumerate(multitask.test_tasks(how_many)):
                    val_loss, val_acc = standard_validate(validate_loader, model, crt_epoch)

                    #  -- Reporting
                    val_info = {"acc": val_acc, "loss": val_loss}

                    new_best_acc, new_best_loss = report.trace_eval(seen, test_task_idx, crt_epoch,
                                                                    val_epoch, all_val_epochs,
                                                                    val_info)
                    all_val_epochs += 1

            val_epoch += 1

            if crt_epoch % save_report_freq == 0:
                report.save()

            all_epochs += 1

        report.save()

        report.finished_training_task(train_task_idx+1, seen)

    report.save()
