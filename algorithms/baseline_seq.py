import torch.nn as nn
import torch.optim as optim
from typing import Type, Callable
import os

# Project imports
from my_types import Args
from multi_task import MultiTask
from utils import standard_train, standard_validate
from reporting import Reporting


def train_sequentially(model_class: Type,
                       get_optimizer: Callable[[nn.Module], optim.Optimizer],
                       multitask: MultiTask,
                       args: Args)-> None:

    print(os.path.basename(__file__))

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

    report = Reporting(args, multitask.get_task_info(), model=model)

    save_report_freq = args.reporting.save_report_freq
    seen = 0
    no_tasks = len(multitask)

    for train_task_idx, data_loaders in enumerate(train_tasks):
        train_loader, validate_loader = data_loaders
        task_name = train_loader.name

        print(f"Training on task {train_task_idx:d}: {task_name:s}.")

        val_epoch = 0

        for crt_epoch in range(epochs_per_task):

            train_loss, train_acc, _ = standard_train(train_loader, model, optimizer, crt_epoch,
                                                      report_freq=batch_report_freq)
            seen += len(train_loader)

            train_info = {"acc": train_acc, "loss": train_loss}
            report.trace_train(seen, train_task_idx, crt_epoch, train_info)

            # Evaluate
            if crt_epoch % eval_freq == 0 or crt_epoch == (epochs_per_task - 1):
                how_many = no_tasks if eval_not_trained else train_task_idx + 1
                for test_task_idx, validate_loader in enumerate(multitask.test_tasks(how_many)):
                    val_loss, val_acc = standard_validate(validate_loader, model, crt_epoch)

                    #  -- Reporting
                    val_info = {"acc": val_acc, "loss": val_loss}

                    new_best_acc, new_best_loss = report.trace_eval(seen, test_task_idx, crt_epoch,
                                                                    val_epoch, val_info)

            val_epoch += 1

            if crt_epoch % save_report_freq == 0:
                report.save()

        report.save()

        report.finished_training_task(train_task_idx+1, seen)

    report.save()
