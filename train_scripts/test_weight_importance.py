"""
    Custom cfg

    test_weight_importance:
        variance_segments: ${eval, torch.arange(0, 10,0.1)}
        no_samples: 10
        max_group_param: 10000

"""
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from typing import Callable, Any, Tuple, Union, Iterator
from termcolor import colored as clr
import torch
import torch.nn.functional as functional
import copy
import sys
import time

# Project imports
from my_types import Args
from multi_task import MultiTask, TaskDataLoader, Batch
from utils.reporting import Reporting
from utils.config_parse import eval_pattern
from utils.util import standard_validate, standard_train


def calc_constraint(model, optimizer, train_loader):
    print("Calculate constraint ... ")
    optimizer.zero_grad()

    for _batch_idx, (data, targets, head_idx) in enumerate(train_loader):
        outputs = model(data, head_idx=head_idx)
        loss = 0
        for output, target in zip(outputs, targets):
            loss += functional.cross_entropy(output, target)
        loss.backward()

    # Calculate empirical fisher constraint
    grad = dict({})
    crt_mode = dict({})

    for name, param in model.named_parameters():
        if param.grad is not None:
            crt_mode[name] = param.detach().clone().view(-1)
            grad[name] = param.grad.detach().pow(2).clone().view(-1)

    return grad, crt_mode


def perturb_network(args: Args, val_loader: TaskDataLoader, model: nn.Module, epoch: int,
                    report_freq: int = 1000):

    print("Perturb network ... ")

    variance_segments = eval_pattern(args.variance_segments)
    no_samples = args.no_samples
    max_group_param = args.max_group_param
    model = copy.deepcopy(model)

    no_segments = len(variance_segments) - 1
    res_size = torch.Size([no_segments, no_samples])

    results = dict()

    no_param = [p.numel() for p in model.parameters() if p.requires_grad]

    val_time = 1
    total_no_param = sum(no_param) * no_segments * no_samples
    done_param = 0
    last_t = time.time()
    first_t = time.time()

    print(f"No parameters: {no_param}; Total: {total_no_param}")

    for name, param in model.named_parameters():
        print(f"\n\nWorking on param: {name} ...")

        if param.requires_grad:

            # Build results tensors
            no_elem = param.nelement()
            param_size = param.size()
            no_dim = len(param_size)

            acc = torch.zeros(param.size())
            acc.unsqueeze_(no_dim).unsqueeze_(no_dim+1)
            acc = acc.expand(param_size + res_size)
            loss = acc.clone()

            acc = acc.view(torch.Size([no_elem]) + res_size)
            loss = loss.view(torch.Size([no_elem]) + res_size)

            sys.stdout.write(f"\rSegment idx: ")
            sys.stdout.flush()

            for segment_idx in range(no_segments):

                # Calculate perturbation values in segment [min_th, max_th] * random_sign
                sample_sign = (torch.rand(no_samples) > 0.5).float() * 2 - 1
                min_th, max_th = variance_segments[segment_idx: segment_idx+2]
                samples = torch.zeros_like(sample_sign).uniform_(min_th, max_th)
                samples.mul_(sample_sign)
                samples = samples.to(param.device)

                for p_ix, perturbation in enumerate(samples):

                    for param_idx in range(min(max_group_param, no_elem)):

                        if (param_idx+1) % report_freq == 0:
                            dur = time.time() - last_t
                            last_t = time.time()
                            remaining_time = (total_no_param - done_param) // report_freq * dur
                            elaps = time.time() - first_t

                            sys.stdout.write(f"\rSegment idx: {segment_idx} / {no_segments};"
                                             f"\t Sample: {p_ix} / {no_samples}"
                                             f"\t Param_idx: {param_idx} / {no_elem}"
                                             f"\t Remaining: {remaining_time:.4f}"
                                             f"\t Elaps: {elaps:.4f}")
                            sys.stdout.flush()

                            done_param += report_freq

                        param.data.view(-1)[param_idx].add_(perturbation)

                        val_loss, val_acc = standard_validate(val_loader, model, epoch)
                        loss[param_idx, segment_idx, p_ix] = val_loss
                        acc[param_idx, segment_idx, p_ix] = val_acc
                        param.data.view(-1)[param_idx].sub_(perturbation)

            acc = acc.view(param_size + res_size)
            loss = loss.view(param_size + res_size)
            results[name] = {"acc": acc, "loss": loss}

    return dict({"results": results, "res_size": res_size, "max_group_param": max_group_param})


def test_weight_importance(init_model: Callable[[Any], nn.Module],
                           get_optimizer: Callable[[nn.Module], optim.Optimizer],
                           multitask: MultiTask,
                           args: Args)-> None:

    print(f"Training {clr('with random random variance on label', attrs=['bold']):s} on all tasks.")

    epochs_per_task = args.train.epochs_per_task
    model_params = args.model

    batch_train_show_freq = args.reporting.batch_train_show_freq

    local_args = args.test_weight_importance

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

        # -- Perturb network
        perturb_info = dict()
        grad, crt_mode = calc_constraint(model, optimizer, train_loader)
        perturb_info["constraint"] = grad
        perturb_info["mode"] = crt_mode
        res_perturb = perturb_network(local_args, validate_loader, model, crt_epoch)
        perturb_info.update(res_perturb)

        report.finished_training_task(task_idx+1, seen, info=perturb_info)

    report.save(final=True)
