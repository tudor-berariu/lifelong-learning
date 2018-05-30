import torch.nn as nn
import torch.optim as optim
from typing import Type, Callable
from termcolor import colored as clr


# Project imports
from my_types import Args
from multi_task import MultiTask
from utils import standard_train, standard_validate
from reporting import Reporting


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
    train_batch_cnt = multitask.merged_tasks_estimated_batches_cnt
    train_task_idx = 0

    # Initialize model & optim
    model: nn.Module = model_class(model_params, in_size, out_size)
    optimizer = get_optimizer(model.parameters())

    report = Reporting(args, multitask.get_task_info(), model=model)
    save_report_freq = args.reporting.save_report_freq

    seen = 0
    val_epoch = 0
    no_tasks = len(multitask)

    for crt_epoch in range(epochs_per_task):
        # TODO Adjust optimizer learning rate

        train_loss, train_acc, train_seen = standard_train(train_loader, model, optimizer,
                                                           crt_epoch, report_freq=batch_report_freq,
                                                           max_batch=train_batch_cnt)
        seen += train_seen

        train_info = {"acc": train_acc, "loss": train_loss}
        report.trace_train(seen, train_task_idx, crt_epoch, train_info)

        for task_idx, validate_loader in enumerate(multitask.test_tasks(no_tasks)):
            val_loss, val_acc = standard_validate(
                validate_loader, model, crt_epoch, report_freq=1)

            #  -- Reporting
            val_info = {"acc": val_acc, "loss": val_loss}

            new_best_acc, new_best_loss = report.trace_eval(seen, task_idx, crt_epoch,
                                                            val_epoch, val_info)

        val_epoch += 1

        if crt_epoch % save_report_freq == 0:
            report.save()

    report.save()
