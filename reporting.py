from typing import Dict, List, NamedTuple, Tuple
import time
import numpy as np
import torch
import os
import subprocess
from comet_ml import Experiment
from termcolor import colored as clr

from my_types import Args, Tasks, Model, LongVector, DatasetTasks, Optional
from multi_task import MultiTask

Accuracy = float
Loss = float

EvalResult = NamedTuple(
    "EvalResult",
    [("task_results", Dict[str, Dict[str, List[float]]]),
     ("dataset_avg", Dict[str, Dict[str, float]]),
     ("global_avg", Dict[str, float])]
)


class TensorboardSummary(object):

    def __init__(self, experiment_name: str, path_to_save: str, auto_start_board: bool):
        from tensorboardX import SummaryWriter

        """
        :param experiment_name:
        :param path_to_save:
        :param auto_start_board: It will start a process that runs Tensorboard on current experiment
        """
        self.save_path = save_path = os.path.join(path_to_save, "tensorboard_logs")

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self._writer = SummaryWriter(log_dir=save_path, comment=experiment_name)

        self.auto_start_board = auto_start_board
        self.board_started = False

    def tick(self, data: List[Tuple[str, Dict, int]]):
        """
        :param data: plot_name, metric_dict, step
        Example:
            [
            "acc", {dataset_1: 33.3, dataset_3: 44.4}, 3
            "loss", {dataset_1: 0.123, dataset_3: 0.33}, 3
            ]
        """
        for plot_name, metric_dict, step in data:

            for k2, v2 in metric_dict.items():
                metric_dict[k2] = float(v2)

            self._writer.add_scalars(plot_name, metric_dict, step)

        if self.auto_start_board and not self.board_started:
            self.start_tensorboard()
            self.board_started = True

    def save_scalars(self, file_name):
        self._writer.export_scalars_to_json(file_name)

    def close(self):
        self._writer.close()

    def start_tensorboard(self, kill_other=True):
        if kill_other:
            os.system("killall -9 tensorboard")

        save_path = self.save_path
        subprocess.Popen(["tensorboard", "--logdir", save_path])


class Reporting(object):

    def __init__(self, args: Args, task_info: List[Dict]):
        self._args = args
        self._task_info = task_info

        self.title = args.title
        self._save_path = os.path.join(args.out_dir, 'reporting')
        self.out_dir = args.out_dir
        self.mode = args.mode
        self.use_tensorboard = args.reporting.plot_tensorboard
        self.use_comet = args.reporting.plot_comet
        self.save_report_trace = args.reporting.save_report_trace

        self.task_idx_to_dataset = task_idx_to_name = dict({x["idx"]: x["dataset_name"]
                                                            for x in task_info})
        self.task_name = dict({x["idx"]: x["name"] for x in task_info})
        self.dataset_task_idx = {d_n: [] for d_n in set(self.task_idx_to_dataset.values())}

        for idx, dataset_name in task_idx_to_name.items():
            self.dataset_task_idx[dataset_name].append(idx)

        self._start_time = time.time()

        self._train_trace = dict({})
        self._eval_trace = dict({})

        # Best Values
        self._best_eval = dict({task_idx: {"acc": {"value": -1, "seen": -1},
                                          "loss": {"value": np.inf, "seen": -1}}
                                for task_idx in task_idx_to_name.keys()})

        self._last_eval = dict({task_idx: {"acc": -1, "seen": -1, "loss":  np.inf}
                                for task_idx in task_idx_to_name.keys()})

        # The following variables will be saved in a dictionary
        self._save_variables = ["_start_time", "_args",
                                "_best_eval", "_last_eval", "_task_info"]
        if self.save_report_trace:
            self._save_variables.extend(["_train_trace", "_eval_trace"])

        self.plot_t: TensorboardSummary = self._init_tensorboard() if self.use_tensorboard else None
        self.plot_c: Experiment = self._init_comet_ml() if self.use_comet else None

    def _init_tensorboard(self):
        # -- Tensorboard SummaryWriter
        plot_t = TensorboardSummary(
                self.title, path_to_save=self.out_dir,
                auto_start_board=self._args.reporting.tensorboard_auto_start
            )
        return plot_t

    def _init_comet_ml(self):
        plot_c = Experiment(api_key="HWzTsfF0Wjk1t45c1T6q9BQMJ",
                            project_name="lifelong-learning",
                            team_name="nemodrivers",
                            log_code=False, log_graph=False,
                            auto_param_logging=False, auto_metric_logging=False,
                            auto_output_logging=False)
        plot_c.log_parameter()
        return plot_c

    def trace_train(self, seen_training: int, task_idx: int, train_epoch: int, info: dict):
        trace = self._train_trace

        if seen_training not in trace:
            trace[seen_training] = dict({})
        if task_idx not in trace[seen_training]:
            trace[seen_training][task_idx] = []

        info["train_epoch"] = train_epoch
        trace[seen_training][task_idx].append(info)

        acc, loss = info["acc"], info["loss"]
        task_name = self.task_name[task_idx]

        # Plot for individual training
        # -- Plot per individual task
        plot_t, plot_c = self.plot_t, self.plot_c
        mode = self.mode
        if mode == "ind":
            if plot_t:
                plot_t.tick([(task_name, {f"loss_train": loss, "acc_train": acc}, train_epoch)])
        elif mode == "sim":
            if plot_t:
                plot_t.tick([(task_name, {f"loss_train": loss, "acc_train": acc}, train_epoch)])

    def trace_eval(self, seen_training: int, task_idx: int, train_epoch: int, val_epoch: int,
                   info: dict) -> Tuple[bool, bool]:

        trace = self._eval_trace

        if seen_training not in trace:
            trace[seen_training] = dict({})
        if task_idx not in trace[seen_training]:
            trace[seen_training][task_idx] = []

        info["train_epoch"] = train_epoch
        info["val_epoch"] = val_epoch
        trace[seen_training][task_idx].append(info)

        best_eval = self._best_eval
        new_best_acc, new_best_loss = False, False

        acc, loss = info["acc"], info["loss"]

        last_eval = self._last_eval
        last_eval[task_idx]["acc"] = acc
        last_eval[task_idx]["loss"] = loss
        last_eval[task_idx]["seen"] = seen_training

        if best_eval[task_idx]["acc"]["value"] < acc:
            best_eval[task_idx]["acc"]["value"] = acc
            best_eval[task_idx]["acc"]["seen"] = seen_training
            new_best_acc = True

        if best_eval[task_idx]["loss"]["value"] > loss:
            best_eval[task_idx]["loss"]["value"] = loss
            best_eval[task_idx]["loss"]["seen"] = seen_training
            new_best_loss = True

        task_name = self.task_name[task_idx]

        # Plot for individual training
        # -- Plot per individual task
        plot_t, plot_c = self.plot_t, self.plot_c
        mode = self.mode
        if mode == "ind":
            if plot_t:
                plot_t.tick([(task_name, {f"loss_eval": loss, "acc_eval": acc}, train_epoch)])

            self._show_task_result(train_epoch, task_name, acc, loss, new_best_acc, new_best_loss)
        elif mode == "sim":
            if plot_t:
                plot_t.tick([(task_name, {f"loss_eval": loss, "acc_eval": acc}, train_epoch)])

            self._show_task_result(train_epoch, task_name, acc, loss, new_best_acc, new_best_loss)

        return new_best_acc, new_best_loss

    def finished_training_task(self, task_idx: int) -> None:
        # TODO Stuff to do when training has finished for a task
        pass

    @property
    def get_dataset_avg(self) -> Dict:
        dataset_task_idx = self.dataset_task_idx
        last_eval = self._last_eval

        dataset_values = {d_n: {"acc": [], "loss": []} for d_n in set(self.dataset_task_idx.keys())}

        for dataset, task_idxs in dataset_task_idx.items():
            for task_idx in task_idxs:
                if last_eval[task_idx]["seen"] != -1:
                    acc, loss = last_eval[task_idx]["acc"], last_eval[task_idx]["loss"]
                    dataset_values[dataset]["acc"].append(acc)
                    dataset_values[dataset]["loss"].append(loss)

        return dataset_values

    @property
    def get_global_avg(self) -> Dict:
        last_eval = self._last_eval

        all_acc = []
        all_loss = []
        for task_idx, value in last_eval.items():
            if value["seen"] != -1:
                all_acc.extend(value["acc"])
                all_loss.extend(value["loss"])

        global_avg = {
            "acc": all_acc,
            "loss": all_loss,
            "mean_acc": np.mean(all_acc),
            "mean_loss": np.mean(all_loss)
        }

        return global_avg

    def save(self):
        save_data = {key: self.__dict__[key] for key in self._save_variables}
        torch.save(save_data, self._save_path)

    @staticmethod
    def _show_task_result(idx: int, task_name: str, acc: float, loss: float,
                          is_acc_better: bool, is_loss_better: bool) -> None:
        msg = f"      [{idx:6}]" +\
              f"[Task {clr(f'{task_name:s}', attrs=['bold']):s}]\t"

        colors = ['white', 'on_magenta'] if is_acc_better else ['yellow']
        msg += f" Accuracy: {clr(f'{acc:5.2f}%', *colors):s}"

        colors = ['white', 'on_magenta'] if is_loss_better else ['yellow']
        msg += f" Loss: {clr(f'{loss:6.4f}', *colors):s}"
        print(msg)
