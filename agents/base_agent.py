import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch import Tensor

from typing import Union, Callable, Any, List, Dict, Iterator, Tuple, Type
import os
import numpy as np
from tabulate import tabulate
from liftoff.config import value_of

# Project imports
from my_types import Args
from multi_task import MultiTask, TaskDataLoader, Batch
from utils.reporting import Reporting
from utils.util import AverageMeter, accuracy


class EmptyScheduler:
    def step(self):
        pass


class BaseAgent(object):
    def __init__(self, get_model: Callable[[Any], nn.Module],
                 get_optimizer: Callable[[nn.Module], optim.Optimizer],
                 multitask: MultiTask, args: Args):

        self.get_model = get_model
        self.get_optimizer = get_optimizer
        self.args = args

        self._args = args
        self.epochs_per_task = args.train.epochs_per_task
        self.max_nan_losses = args.train.max_nan_loss
        self.early_stop = value_of(args.train, "early_stop", np.inf)
        self.multitask = multitask
        self.device = torch.device(args.device)

        self.eval_freq = args.reporting.eval_freq
        self.eval_not_trained = args.reporting.eval_not_trained
        self.save_report_freq = args.reporting.save_report_freq

        self._train_batch_save_freq = args.reporting.batch_train_save_freq
        self._train_batch_show_freq = args.reporting.batch_train_show_freq
        self._eval_batch_save_freq = args.reporting.batch_eval_save_freq
        self._eval_batch_show_freq = args.reporting.batch_eval_show_freq

        self._train_batch_report = (self._train_batch_save_freq + self._train_batch_show_freq) > 0
        self._eval_batch_report = (self._eval_batch_save_freq + self._eval_batch_show_freq) > 0

        self.in_size = in_size = multitask.in_size
        self.out_size = out_size = multitask.out_size
        self.no_tasks: int = len(multitask)
        self.total_no_of_epochs = self.epochs_per_task * self.no_tasks

        # Initialize model & optim
        self._model: nn.Module = None
        self._optimizer: optim.Optimizer = None
        self._init_model(get_model, get_optimizer, args, in_size, out_size)

        self.train_tasks = multitask.train_tasks()

        model_summary = self.get_model_summary()
        self.report = Reporting(args, multitask.get_task_info(), model_summary=model_summary,
                                files_to_save=[os.path.abspath(__file__)])

        # Local variables
        self.crt_train_info = dict({})
        self.crt_eval_info = dict({})
        self.seen: int = 0
        self.all_epochs: int = 0
        self.all_eval_epochs: int = 0
        self.all_eval_ind_epochs: int = 0
        self.crt_task_epoch: int = -1
        self.crt_task_idx: int = 0
        self.crt_data_loaders: Iterator[Tuple[TaskDataLoader, TaskDataLoader]] = None
        self.crt_task_name: str = None
        self.crt_eval_epoch: int = 0
        self.crt_nan_losses = 0

        self.stop_training = False
        self.stop_training_info = dict()

        self._batch_aux_losses = {}

    def _init_model(self, get_model: Type,
                    get_optimizer: Callable[[nn.Module], optim.Optimizer],
                    args: Args, in_size: torch.Size, out_size: List[int]) -> None:

        self._model: nn.Module = get_model(args.model, in_size, out_size)
        self._optimizer = get_optimizer(self._model.parameters())

        optim_args = args.train._optimizer
        if hasattr(optim_args, "lr_decay"):
            step = optim_args.lr_decay.step
            gamma = optim_args.lr_decay.gamma
            milestones = list(range(step, self.total_no_of_epochs, step))
            self.scheduler = optim.lr_scheduler.MultiStepLR(self._optimizer,
                                                            milestones=milestones,
                                                            gamma=gamma)
        else:
            self.scheduler = EmptyScheduler()

    def get_model_summary(self) -> Dict:
        if isinstance(self._model, nn.Module):
            return {"summary": self._model.__str__()}
        return {}

    def train_sequentially(self):

        self._start_experiment()  # TEMPLATE

        # Load general Classes
        report = self.report
        multitask = self.multitask
        seen = 0

        # Loop over tasks
        for self.crt_task_idx, self.crt_data_loaders in enumerate(self.train_tasks):
            early_stop_task = False
            eval_no_improvement = 0

            train_task_idx, data_loaders = self.crt_task_idx, self.crt_data_loaders

            train_loader, val_loader = data_loaders
            task_name = self.crt_task_name = train_loader.name
            self.crt_eval_epoch = 0

            print(f"Training on task {train_task_idx:d}: {task_name:s}.")

            self._start_train_task()  # TEMPLATE

            # Train each task for a number of epochs
            for self.crt_task_epoch in range(self.epochs_per_task):
                crt_epoch = self.crt_task_epoch

                # Train task 1 epoch
                train_loss, train_acc, seen, info = self._train_epoch(train_task_idx, train_loader,
                                                                      crt_epoch)

                # Get information to feed to reporting agent
                train_info = {"acc": train_acc, "loss": train_loss}
                train_info.update(info)
                report.trace_train(self.seen, train_task_idx, crt_epoch,
                                   self.all_epochs, train_info)

                # Evaluate
                if crt_epoch % self.eval_freq == 0 or crt_epoch == (self.epochs_per_task - 1):
                    eval_epoch = self.crt_eval_epoch

                    # Validate on first 'how_many' tasks
                    how_many = self.no_tasks if self.eval_not_trained else train_task_idx + 1
                    new_best_acc_cnt = new_best_loss_cnt = 0
                    for val_task_idx, val_loader in enumerate(multitask.test_tasks(how_many)):
                        val_loss, val_acc, info = self._eval_task(val_task_idx, val_loader,
                                                                  crt_epoch, eval_epoch)

                        #  -- Reporting
                        val_info = {"acc": val_acc, "loss": val_loss}
                        val_info.update(info)

                        new_best_acc, new_best_loss = report.trace_eval(self.seen, val_task_idx,
                                                                        crt_epoch, eval_epoch,
                                                                        self.all_eval_ind_epochs,
                                                                        val_info)
                        new_best_acc_cnt += new_best_acc
                        new_best_loss_cnt += new_best_loss

                        # Early task stop
                        if val_task_idx == train_task_idx:
                            if new_best_acc + new_best_loss > 0:
                                eval_no_improvement = 0
                            else:
                                eval_no_improvement += 1
                                if eval_no_improvement > self.early_stop:
                                    early_stop_task = True

                        self.all_eval_ind_epochs += 1

                    self.crt_eval_epoch += 1
                    self.all_eval_epochs += 1
                    self.scheduler.step()

                if crt_epoch % self.save_report_freq == 0:
                    report.save()

                if self.stop_training or early_stop_task:
                    print(f"[TRAIN] Early stop task. stop_training: {self.stop_training}"
                          f" early_stop_task: {early_stop_task}")
                    break

                self.all_epochs += 1

            self._end_train_task()

            report.save()

            report.finished_training_task(train_task_idx+1, self.seen)

            if self.stop_training:
                break

        self._end_experiment()  # TEMPLATE

        report.save(final=True)

    def batch_update_auxiliary_losses(self, info: dict) -> None:
        for key, value in info.items():
            if key.startswith("loss_"):
                meter = self._batch_aux_losses.setdefault(key, AverageMeter())
                meter.update(value)

    def batch_print_aux_losses(self) -> None:
        table = []
        for key, meter in self._batch_aux_losses.items():
            table.append([key[5:], meter.val, meter.avg])
        print(tabulate(table, headers=["Loss", "Crt.", "Avg."]))

    def _train_epoch(self, task_idx: int, train_loader: Union[TaskDataLoader, Iterator[Batch]],
                     epoch: int, max_batch: int = np.inf) -> Tuple[float, float, int, Dict]:

        self.crt_train_info = info = dict({})

        self._start_train_epoch()

        report_or_not = self._train_batch_report
        print_freq = self._train_batch_show_freq
        report_freq = self._train_batch_save_freq
        report = self.report

        last_batch = len(train_loader) - 1

        losses = AverageMeter()
        acc = AverageMeter()
        correct_cnt = 0
        seen_epoch = 0

        self.train()
        for batch_idx, (data, targets, head_idx) in enumerate(train_loader):
            if batch_idx > max_batch:
                break

            self._start_train_task_batch()
            outputs, loss, info_batch = self._train_task_batch(batch_idx, data, targets, head_idx)
            self._end_train_task_batch(outputs, loss, info_batch)

            info.update(info_batch)
            self.batch_update_auxiliary_losses(info_batch)

            (top1, correct), = accuracy(outputs, targets)
            correct_cnt += correct

            seen_epoch += data.size(0)
            self.seen += data.size(0)
            acc.update(top1, data.size(0))
            losses.update(loss.item(), data.size(0))

            if report_or_not:
                if report_freq > 0:
                    if (batch_idx + 1) % report_freq == 0 or batch_idx == last_batch:
                        report.trace_train_batch(self.seen, task_idx, epoch, info)

                if print_freq > 0:
                    if (batch_idx + 1) % print_freq == 0 or batch_idx == last_batch:
                        print(f'\t\t[Train] [Epoch: {epoch:3}] [Batch: {batch_idx:5}]:\t '
                              f'[Loss] crt: {losses.val:3.4f}  avg: {losses.avg:3.4f}\t'
                              f'[Accuracy] crt: {acc.val:3.2f}  avg: {acc.avg:.2f}')
                        self.batch_print_aux_losses()

            if self.stop_training:
                break

        self._end_train_epoch()

        return losses.avg, correct_cnt / float(seen_epoch), seen_epoch, info

    def _train_task_batch(self, batch_idx: int, data: Tensor, targets: List[Tensor],
                          head_idx: Union[int, Tensor])-> Tuple[List[Tensor], Tensor, Dict]:

        self._optimizer.zero_grad()
        outputs = self._model(data, head_idx=head_idx)

        loss = torch.tensor(0., device=self.device)
        for out, t in zip(outputs, targets):
            loss += functional.cross_entropy(out, t)

        loss.backward()
        self._optimizer.step()

        return outputs, loss, dict({})

    def _eval_task(self, task_idx: int, val_loader: TaskDataLoader,
                   train_epoch: int, val_epoch: int) -> Tuple[float, float, Dict]:
        self.crt_eval_info = info = dict({})

        self._start_eval_task()  # TEMPLATE

        report_or_not = self._eval_batch_report
        print_freq = self._eval_batch_show_freq
        report_freq = self._eval_batch_save_freq
        report = self.report

        last_batch = len(val_loader) - 1
        losses = AverageMeter()
        acc = AverageMeter()
        correct_cnt = 0
        seen = self.seen
        seen_eval = 0

        with torch.no_grad():
            for batch_idx, (data, targets, head_idx) in enumerate(val_loader):
                outputs, loss, info_batch = self._eval_task_batch(batch_idx, data, targets,
                                                                  head_idx)
                info.update(info_batch)

                (top1, correct), = accuracy(outputs, targets)
                correct_cnt += correct

                seen_eval += data.size(0)
                acc.update(top1, data.size(0))
                losses.update(loss.item(), data.size(0))

                if report_or_not:
                    if report_freq > 0:
                        if (batch_idx + 1) % report_freq == 0 or batch_idx == last_batch:
                            report.trace_eval_batch(seen, task_idx, train_epoch, val_epoch, info)

                    if print_freq > 0:
                        if (batch_idx + 1) % print_freq == 0 or batch_idx == last_batch:
                            print(f'\t\t[Eval] [Epoch: {train_epoch:3}] [Batch: {batch_idx:5}]:\t '
                                  f'[Loss] crt: {losses.val:3.4f}  avg: {losses.avg:3.4f}\t'
                                  f'[Accuracy] crt: {acc.val:3.2f}  avg: {acc.avg:.2f}')

            self._end_eval_task()  # TEMPLATE

            return losses.avg, correct_cnt / float(seen_eval), info

    def _eval_task_batch(self, batch_idx: int, data: Tensor, targets: List[Tensor],
                         head_idx: Union[int, Tensor]) -> Tuple[List[Tensor], Tensor, Dict]:

        outputs = self._model(data, head_idx=head_idx)

        loss = torch.tensor(0., device=self.device)
        for out, t in zip(outputs, targets):
            loss += functional.cross_entropy(out, t)

        return outputs, loss, dict({})

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    def _start_experiment(self):
        pass

    def _end_experiment(self):
        pass

    def _start_train_task(self):
        pass

    def _end_train_task(self):
        pass

    def _start_train_epoch(self):
        pass

    def _end_train_epoch(self):
        pass

    def _start_eval_task(self):
        pass

    def _end_eval_task(self):
        pass

    def _start_train_task_batch(self):
        pass

    def _end_train_task_batch(self, outputs: Tuple[List[Tensor]], loss: Tensor, info: Dict):
        self._register_train_task_batch_loss(loss, info)

    def _register_train_task_batch_loss(self, loss: Tensor, info: Dict):
        if torch.isnan(loss):
            self.crt_nan_losses += 1
            if self.crt_nan_losses > self.max_nan_losses:
                self.stop_training = True
                self.stop_training_info["max_nan_loss_reached"] = self.crt_nan_losses
                info.update({"stop_training_info": self.stop_training_info})
        else:
            self.crt_nan_losses = 0
