from typing import Dict, List, NamedTuple, Tuple, Union
import time
import numpy as np
import torch
import os
from termcolor import colored as clr
from shutil import copyfile
from liftoff.config import namespace_to_dict
import subprocess
from argparse import Namespace

from my_types import Args
from .utils import get_ip, redirect_std, repair_std, get_utc_time


Accuracy = float
Loss = float

REMOTE_IP = "141.85.232.73"
REMOTE_HOST = "tempuser@141.85.232.73"
SERVER_eFOLDER = "/home/tempuser/workspace/andrei/lifelong_tmp_data/"
SERVER_PYTHON = "/home/tempuser/anaconda3/envs/andreiENV/bin/python"
SERVER_SCRIPT = "/home/tempuser/workspace/andrei/lifelong-learning/utils/upload_to_elasticsearch.py"

BIG_DATA_KEYS = ["_train_trace", "_eval_trace", "_task_train_tick"]

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
        self.save_path = save_path = os.path.join(
            path_to_save, "tensorboard_logs")

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self._writer = SummaryWriter(
            log_dir=save_path, comment=experiment_name)

        self.auto_start_board = auto_start_board
        self.board_started = False

    def tick(self, data: List[Tuple[str, Dict, int]]):
        """
        :param data: plot_name, metric_dict, step
        Example:
            [
            "mnist_p0/acc", {dataset_1: 33.3, dataset_3: 44.4}, 3
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

    @property
    def tb(self):
        return self._writer


class Reporting(object):

    def __init__(self, args: Args, task_info: List[Dict], model_summary: Dict = dict(),
                 files_to_save: List[str] = list()):

        self._args = namespace_to_dict(args)
        self._task_info = task_info

        self.title = args.title
        self._save_path = os.path.join(args.out_dir, 'reporting.pkl')
        self.out_dir = args.out_dir
        self.mode = args.mode
        self.use_tensorboard = args.reporting.plot_tensorboard
        self.use_comet = args.reporting.plot_comet
        self.save_report_trace = args.reporting.save_report_trace
        self.min_save = min_save = args.reporting.min_save
        self.push_to_server = args.reporting.push_to_server

        # Register model summary
        self._model_summary = None
        self.register_model(model_summary)

        # Folders where Server side reports should be temporarily be moved
        base_res_fld = "results/"
        local_efolder = base_res_fld + "tmp_efolder_data"
        if not os.path.isdir(local_efolder):
            assert os.path.isdir("results"), f"There is no base results folder, expected: " \
                                             f"{base_res_fld}"
            os.mkdir(local_efolder)
        assert os.path.isdir(local_efolder), f"There is no temporary local efolder {local_efolder}"
        self.local_efolder = local_efolder

        # Copy files files_to_save
        if not min_save:
            for file_path in files_to_save:
                copyfile(file_path, os.path.join(self.out_dir,
                                                 f"{os.path.basename(file_path)}_script"))

        # Should read baseline for each task (individual/ simultanous)
        self.task_base_ind = dict({x["idx"]: x["best_individual"] for x in task_info})
        self.task_base_sim = dict({x["idx"]: x["best_simultaneous"] for x in task_info})

        self.task_idx_to_dataset = task_idx_to_name = dict({x["idx"]: x["dataset_name"]
                                                            for x in task_info})
        self.no_tasks = len(task_idx_to_name)
        self.task_name = dict({x["idx"]: x["name"] for x in task_info})
        self.dataset_task_idx = {d_n: []
                                 for d_n in set(self.task_idx_to_dataset.values())}

        for idx, dataset_name in task_idx_to_name.items():
            self.dataset_task_idx[dataset_name].append(idx)

        self._max_seen_train = -1
        self._max_seen_eval = -1
        self._trained_before_eval = []
        self._has_evaluated = False

        self._start_timestamp = time.time()
        self._start_time = get_utc_time()

        self._train_trace = dict({})
        self._eval_trace = dict({})

        # Best Values
        self._best_eval = dict({task_idx: {"acc": {"value": -1, "seen": -1},
                                           "loss": {"value": np.inf, "seen": -1}}
                                for task_idx in task_idx_to_name.keys()})
        self._last_eval = dict({task_idx: {"acc": -1, "seen": -1, "loss":  np.inf}
                                for task_idx in task_idx_to_name.keys()})

        self._best_train = self._best_eval.copy()
        self._last_train = self._last_eval.copy()

        self._task_train_tick: List[Dict] = []
        self._last_eval_data: Dict = dict()
        self._eval_metrics: Dict = dict({
            "score_new_raw": [],
            "score_base_raw": [],
            "score_all_raw": [],
            "score_all_last_raw": [],
            "score_new": -1,
            "score_base": -1,
            "score_all_all": -1,
        })

        # The following variables will be saved in a dictionary
        self.big_data = ["_train_trace", "_eval_trace", "_task_train_tick"]
        self._save_variables = ["_start_time", "_start_timestamp", "_args",
                                "_best_eval", "_last_eval", "_task_info",
                                "_eval_metrics", "_model_summary", "_last_eval_data"]
        if self.save_report_trace:
            self._save_variables.extend(self.big_data)

        # Plot data
        self.plot_t: TensorboardSummary = self._init_tensorboard(
        ) if self.use_tensorboard and not min_save else None
        # self.plot_c: Experiment = self._init_comet_ml() if self.use_comet else None
        self.plot_c = None

        self.es = None

    def _init_tensorboard(self):
        # -- Tensorboard SummaryWriter
        plot_t = TensorboardSummary(
            self.title, path_to_save=self.out_dir,
            auto_start_board=self._args["reporting"]["tensorboard_auto_start"]
        )
        return plot_t

    def register_model(self, model_summary: Dict):
        if self.min_save:
            return

        if self._model_summary:
            op = "a"
        else:
            self._model_summary = []
            op = "w"

        self._model_summary.append(model_summary)

        with open(os.path.join(self.out_dir, "model_summary"), op) as f:
            s = "=" * 40 + f"{len(self._model_summary):4}   " + "=" * 40 + "\n"
            f.writelines([s, "\n"])
            for k, v in self._model_summary[-1].items():
                f.write(str(k) + " :: \n")
                f.write(str(v))
            f.writelines(["\n", "\n"])

    @property
    def tb(self):
        return self.plot_t

    def _init_comet_ml(self):
        from comet_ml import Experiment
        plot_c = Experiment(api_key="HWzTsfF0Wjk1t45c1T6q9BQMJ",
                            project_name="lifelong-learning",
                            team_name="nemodrivers",
                            log_code=False, log_graph=False,
                            auto_param_logging=False, auto_metric_logging=False,
                            auto_output_logging=False)
        plot_c.log_parameter()
        return plot_c

    def trace_train(self, seen_training: int, task_idx: int, train_epoch: int, info: dict):
        info = info.copy()

        trace = self._train_trace
        self._max_seen_train = seen_training

        if self._has_evaluated:
            self._has_evaluated = False
            self._trained_before_eval.clear()

        if task_idx not in self._trained_before_eval:
            self._trained_before_eval.append(task_idx)

        if seen_training not in trace:
            trace[seen_training] = dict({})
        if task_idx not in trace[seen_training]:
            trace[seen_training][task_idx] = []

        info["train_epoch"] = train_epoch
        trace[seen_training][task_idx].append(info)

        acc, loss = info["acc"], info["loss"]
        task_name = self.task_name[task_idx]

        # Update best
        new_best_acc, new_best_loss = self.update_best(info, self._last_train, self._best_train,
                                                       task_idx, seen_training)

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
        elif mode == "seq":
            if plot_t:
                plot_t.tick([(task_name, {f"loss_train": loss, "acc_train": acc}, train_epoch)])

    def trace_train_batch(self, seen_training: int, task_idx: int, train_epoch: int, info: dict):
        pass

    def trace_eval(self, seen_training: int, task_idx: int, train_epoch: int, val_epoch: int,
                   info: dict) -> Tuple[bool, bool]:
        info = info.copy()

        self._has_evaluated = True
        trace = self._eval_trace
        self._max_seen_eval = seen_training

        if seen_training not in trace:
            trace[seen_training] = dict({})
        if task_idx not in trace[seen_training]:
            trace[seen_training][task_idx] = []

        info["train_epoch"] = train_epoch
        info["val_epoch"] = val_epoch
        trace[seen_training][task_idx].append(info)

        acc, loss = info["acc"], info["loss"]

        # Update best
        new_best_acc, new_best_loss = self.update_best(info, self._last_eval, self._best_eval,
                                                       task_idx, seen_training)

        task_name = self.task_name[task_idx]
        crt_training = task_idx in self._trained_before_eval

        # Plot for individual training
        # -- Plot per individual task
        plot_t, plot_c = self.plot_t, self.plot_c
        mode = self.mode
        if mode == "ind":
            if plot_t:
                plot_t.tick([(task_name, {f"loss_eval": loss, "acc_eval": acc}, train_epoch)])

            self._show_task_result(train_epoch, task_name, acc, loss,
                                   new_best_acc, new_best_loss, crt_training)

        elif mode == "sim":
            if plot_t:
                plot_t.tick([(task_name, {f"loss_eval": loss, "acc_eval": acc}, train_epoch)])

            self._show_task_result(train_epoch, task_name, acc, loss, new_best_acc,
                                   new_best_loss, crt_training)

        elif mode == "seq":
            no_tasks = self.no_tasks
            if plot_t:
                # plot_t.tick([(task_name, {f"loss_eval": loss, "acc_eval": acc}, train_epoch)])
                level = no_tasks - task_idx
                plot_t.tick([("global/multi",
                              {f"{task_name}_loss_eval": level + loss,
                               f"{task_name}_acc_eval": level + acc},
                              seen_training)])

            self._show_task_result(train_epoch, task_name, acc, loss,
                                   new_best_acc, new_best_loss, crt_training)

        return new_best_acc, new_best_loss

    def trace_eval_batch(self, seen_training: int, task_idx: int,
                         train_epoch: int, val_epoch: int, info: dict):
        pass

    @staticmethod
    def update_best(info: Dict, last: Dict, best: Dict,
                    task_idx: int, seen_training: int) -> Tuple[bool, bool]:

        # Calculate best loss and eval
        new_best_acc, new_best_loss = False, False
        acc, loss = info["acc"], info["loss"]

        last[task_idx]["seen"] = seen_training

        if best[task_idx]["acc"]["value"] < acc:
            best[task_idx]["acc"]["value"] = acc
            best[task_idx]["acc"]["seen"] = seen_training
            best[task_idx]["acc"]["info"] = info.copy()
            new_best_acc = True

        if best[task_idx]["loss"]["value"] > loss:
            best[task_idx]["loss"]["value"] = loss
            best[task_idx]["loss"]["seen"] = seen_training
            best[task_idx]["loss"]["info"] = info.copy()
            new_best_loss = True

        info["new_best_acc"], info["new_best_loss"] = new_best_acc, new_best_loss

        last[task_idx].update(info.copy())
        return new_best_acc, new_best_loss

    def finished_training_task(self, no_trained_tasks: int, seen: int) -> None:
        print(''.join("" * 79))

        print(f"Finished training {clr(f'{no_trained_tasks}', attrs=['bold']):s}"
              f"\t seen: {seen} images")

        last_eval = self._last_eval
        task_train_tick = self._task_train_tick

        eval_data = dict({})

        task_eval_data = dict({})

        not_evaluated = []
        for task_idx, value in last_eval.items():
            if value["seen"] == seen:
                task_eval_data[task_idx] = value
            else:
                not_evaluated.append(task_idx)

        tasks_idxs = []
        all_acc = []
        all_loss = []
        for key in sorted(task_eval_data):
            tasks_idxs.append(key)
            all_acc.append(task_eval_data[key]["acc"])
            all_loss.append(task_eval_data[key]["loss"])

        eval_data["task"] = task_eval_data

        eval_data["global_avg"] = {
            "mean_acc_all": np.mean(all_acc),
            "mean_loss_all": np.mean(all_loss),
            "mean_acc_trained": np.mean(all_acc[:no_trained_tasks]),
            "mean_loss_trained": np.mean(all_loss[:no_trained_tasks]),
        }

        eval_data["global"] = {
            "tasks_idxs": tasks_idxs,
            "acc": all_acc,
            "loss": all_loss
        }

        task_train_tick.append(eval_data)

        # Calculate metrics
        eval_metrics = self._eval_metrics
        eval_metrics["score_new_raw"].append(task_eval_data[no_trained_tasks-1]["acc"])
        if 0 in task_eval_data:
            eval_metrics["score_base_raw"].append(task_eval_data[0]["acc"])
        else:
            eval_metrics["score_base_raw"].append(eval_metrics["score_base_raw"][-1])

        score_all = 0
        score_all_raw = []
        base = self.task_base_ind
        base_ordered = [base[i] for i in range(no_trained_tasks)]

        for key in sorted(task_eval_data):
            if key < no_trained_tasks:
                score_all += task_eval_data[key]["acc"] / base[key]
                score_all_raw.append(task_eval_data[key]["acc"])

        score_all = score_all / float(no_trained_tasks)

        eval_metrics["score_all_raw"].append(score_all_raw)
        eval_metrics["score_all_last_raw"].append(score_all)

        score_new_s = self.norm_scores(eval_metrics["score_new_raw"], base_ordered)
        score_base_s = self.norm_scores(eval_metrics["score_base_raw"], base_ordered)
        score_all_s = eval_metrics["score_all_last_raw"]

        scores = self.calc_scores(eval_metrics, base)
        eval_metrics.update(scores)

        score_new = eval_metrics["score_new"]
        score_base = eval_metrics["score_base"]
        score_all = eval_metrics["score_all"]

        def l_msg(ls: List[float]):
            return "".join([f"{x:3.6f}   " for x in ls])

        idx = no_trained_tasks - 1
        print(f"\t[{idx:3}] score New's  (n):   [{score_new:3.6f}]    [{l_msg(score_new_s)}]")
        print(f"\t[{idx:3}] score Base's (n):   [{score_base:3.6f}]    [{l_msg(score_base_s)}")
        print(f"\t[{idx:3}] score All's  (n):   [{score_all:3.6f}]    [{l_msg(score_all_s)}]")

        # Plot
        mode = self.mode
        plot_t = self.plot_t
        if mode == "seq":
            if plot_t:
                plot_t.tick([("global/average", eval_data["global_avg"], no_trained_tasks)])
                plot_t.tick([("global/average", scores, no_trained_tasks)])

                # Draw vertical line
                plot_t.tick([("global/multi", {f"marker_{task_idx}": 0}, seen)])
                plot_t.tick([("global/multi", {f"marker_{task_idx}": self.no_tasks+1}, seen)])

        print(''.join("" * 79))

    @staticmethod
    def norm_scores(scores: List[float], base: List[float]):
        return [s / b for s, b in zip(scores, base)]

    @staticmethod
    def calc_scores(scores: Dict, base_scores: Dict):
        score_new_raw = scores["score_new_raw"]
        score_base_raw = scores["score_base_raw"]
        score_all_raw = scores["score_all_raw"]
        no_tasks = len(score_new_raw)

        score_new, score_base, score_all = 0, 0, []
        for ix in range(no_tasks):
            score_new += score_new_raw[ix] / base_scores[ix]
            score_base += score_base_raw[ix] / base_scores[ix]
            score_all.append(np.mean([v / base_scores[i] for i, v in enumerate(score_all_raw[ix])]))

        score_new = score_new / float(no_tasks)
        score_base = score_base / float(no_tasks)
        score_last = score_all[-1]
        score_all = np.mean(score_all)

        scores = {
            "score_new": score_new,
            "score_base": score_base,
            "score_all": score_all,
            "score_last": score_last
        }
        return scores

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
                all_acc.append(value["acc"])
                all_loss.append(value["loss"])

        global_avg = {
            "acc": all_acc,
            "loss": all_loss,
            "mean_acc": np.mean(all_acc),
            "mean_loss": np.mean(all_loss)
        }

        return global_avg

    def save(self, final=False):
        save_data = {key: self.__dict__[key] for key in self._save_variables}
        if final:
            save_data["_end_time"] = get_utc_time()
        torch.save(save_data, self._save_path)
        if final:
            self.experiment_finished(save_data, ignore_keys=self.big_data,
                                     local_efolder=self.local_efolder,
                                     push_to_server=self.push_to_server)

    @staticmethod
    def _show_task_result(idx: int, task_name: str, acc: float, loss: float,
                          is_acc_better: bool, is_loss_better: bool,
                          crt_trained: bool) -> None:
        msg = f"\t[{idx:3}] " +\
              f"[Task {clr(f'{task_name:s}', attrs=['bold']):s}]\t"

        colors = ['white', 'on_magenta'] if is_acc_better else ['yellow']
        msg += f" Accuracy: {clr(f'{acc:5.2f}%', *colors):s}"

        colors = ['white', 'on_magenta'] if is_loss_better else ['yellow']
        msg += f" Loss: {clr(f'{loss:6.4f}', *colors):s}"

        if crt_trained:
            msg += f" {clr('*', 'red', attrs=['bold'])}"

        print(msg)

    @staticmethod
    def experiment_finished(save_data: Union[Dict, str], ignore_keys: List[str] = BIG_DATA_KEYS,
                            local_efolder: str = "results/tmp_efolder_data",
                            push_to_server: bool = True):

        save_data_path = None
        if isinstance(save_data, str):
            print(f"Load from disk results pkl. ({save_data})")
            save_data_path = save_data
            save_data = torch.load(save_data_path)

        save_data = save_data.copy()

        for k in ignore_keys:
            save_data.pop(k, None)

        data = Reporting.fix_older_data(save_data)

        start_time = data["start_timestamp"]

        # Move data to local temporary folder
        basename = str(start_time).replace(".", "_") + "_"
        filename = basename + "edata.pkl"
        data_filepath = os.path.join(local_efolder, filename)
        torch.save(data, data_filepath)

        print(f"eData data moved to {data_filepath}")

        if push_to_server:
            import pickle
            import subprocess
            from utils.pid_wait import wait_pid
            import shutil

            # -- Redirect all std out & errors to tmp_edata file
            out_filepath = os.path.join(local_efolder, basename + "_std_out_err.txt")

            fsock, old_stdout, old_stderr = redirect_std(out_filepath)

            # Get ip
            local_ip = get_ip()

            # -- Try to move eData file to server
            server_path = SERVER_eFOLDER + filename
            if local_ip != REMOTE_HOST:
                p = subprocess.Popen(["scp", data_filepath, f"{REMOTE_HOST}:{server_path}"],
                                     stdout=fsock, stderr=fsock)
                sts = wait_pid(p.pid, timeout=120)
                if sts == 0:
                    print(f"[eData] Success in moving eData to server. (RESPONSE: {sts})")
                else:
                    print(f"[eData] ERROR in moving eData to server. (RESPONSE: {sts})")

                    repair_std(out_filepath, fsock, old_stdout, old_stderr)

                    return 1
            else:
                shutil.copy(data_filepath, server_path)

            # -- Check if file was moved so as to delete local file
            def parse_file_size(res, cmd: str):
                fsize = -1

                if hasattr(res, "strip"):
                    res = res.strip().decode("utf-8")
                else:
                    print(f"[eData] res is not string {res}")
                    return fsize

                if res.replace('.', '', 1).isnumeric():
                    fsize = float(res)
                else:
                    print(f"[eData] Wrong _{cmd}_ file_size {res}")

                return fsize

            remote_cmd = ""
            if local_ip != REMOTE_HOST:
                remote_cmd = f"ssh {REMOTE_HOST} "

            # Get local size
            p = subprocess.Popen(f"wc -c {data_filepath} | awk \'{{print $1}}\'",
                                 shell=True, stdout=subprocess.PIPE, stderr=fsock)
            sts = wait_pid(p.pid, timeout=20)
            result = p.communicate()[0]
            local_file_size = parse_file_size(result, "local")

            # Get remote size
            p = subprocess.Popen(remote_cmd + f"wc -c {server_path} | awk \'{{print $1}}\'",
                                 shell=True, stdout=subprocess.PIPE, stderr=fsock)
            sts = wait_pid(p.pid, timeout=20)
            result = p.communicate()[0]
            remote_file_size = parse_file_size(result, "remote")

            repair_std(out_filepath, fsock, old_stdout, old_stderr)

            if local_file_size == remote_file_size and local_file_size > 0:
                print(f"[eData] We consider file copied successfully")
                # Delete local files
                os.remove(data_filepath)
                os.remove(out_filepath)
            else:
                print(f"[eData] ERROR something went wrong with copying the file")
                return 2

            # Try to run server side script, that uplods data to elasticsearch

            # p = subprocess.Popen(remote_cmd + f"nohup {SERVER_PYTHON} {SERVER_SCRIPT} "
            #                                   f"{server_path} &", shell=True)
            p = subprocess.Popen(remote_cmd + f"{SERVER_PYTHON} {SERVER_SCRIPT} {server_path}",
                                 shell=True)
            sts = wait_pid(p.pid, timeout=120)

            print(f"SERVER_SCRIP Response {sts}")

            print(f"Seems ok: {save_data_path}")
        return 0

    @staticmethod
    def fix_older_data(data: Dict):
        import datetime
        # -- Fix older data
        # Bad for elasticsearch
        if isinstance(data["_args"], Namespace):
            data["_args"] = namespace_to_dict(data["_args"])
            data["_args"]["model"]["_conv"] = str(data["_args"]["model"]["_conv"])

        if "_start_timestamp" not in data:
            data["_start_timestamp"] = data["_start_time"]

        if isinstance(data["_start_time"], float):
            data["_start_time"] = datetime.datetime.utcfromtimestamp(data["_start_time"])

        new_data = dict({})
        for k, v in data.items():
            while k[0] == "_":
                k = k[1:]
            new_data[k] = v

        return new_data


if __name__ == "__main__":
    import sys
    import os

    cwd = os.getcwd()
    print(cwd)

    Reporting.experiment_finished(sys.argv[1:])
