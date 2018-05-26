from copy import deepcopy
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from termcolor import colored as clr
import time

from my_types import Args, Tasks, Model, LongVector, DatasetTasks

Accuracy = float
Loss = float

EvalResult = NamedTuple(
    "EvalResult",
    [("task_results", Dict[str, Dict[str, List[float]]]),
     ("dataset_avg", Dict[str, Dict[str, float]]),
     ("global_avg", Dict[str, float])]
)


def update_results(results: EvalResult,
                   best: Optional[EvalResult]) -> Tuple[EvalResult, bool]:
    from operator import lt, gt

    if not best:
        return deepcopy(results), True
    changed = False

    for dataset, dataset_results in results.task_results.items():
        best_results = best.task_results[dataset]

        acc_pairs = zip(dataset_results["acc"], best_results["acc"])
        changed = changed or any(map(lambda p: gt(*p), acc_pairs))
        acc_pairs = zip(dataset_results["acc"], best_results["acc"])
        best_results["acc"] = [max(*p) for p in acc_pairs]

        loss_pairs = zip(dataset_results["loss"], best_results["loss"])
        changed = changed or any(map(lambda p: lt(*p), acc_pairs))
        loss_pairs = zip(dataset_results["loss"], best_results["loss"])
        best_results["loss"] = [min(*p) for p in loss_pairs]

        crt_avg = results.dataset_avg[dataset]
        best_avg = best.dataset_avg[dataset]

        changed = changed or gt(crt_avg["acc"], best_avg["acc"])
        best_avg["acc"] = max(crt_avg["acc"], best_avg["acc"])
        changed = changed or lt(crt_avg["loss"], best_avg["loss"])
        best_avg["loss"] = min(crt_avg["loss"], best_avg["loss"])

    crt_g_avg = results.global_avg
    best_g_avg = best.global_avg

    changed = changed or gt(crt_g_avg["acc"], best_g_avg["acc"])
    best_g_avg["acc"] = max(crt_g_avg["acc"], best_g_avg["acc"])
    changed = changed or lt(crt_g_avg["loss"], best_g_avg["loss"])
    best_g_avg["loss"] = min(crt_g_avg["loss"], best_g_avg["loss"])

    return best, changed


def show_results(seen: int, results: EvalResult,
                 best: Optional[EvalResult]) -> None:
    print(''.join("" * 79))
    print(f"Evaluation {clr(f'after {seen:d} examples', attrs=['bold']):s}:")

    for dataset, dataset_results in results.task_results.items():

        accuracies = dataset_results["acc"]
        losses = dataset_results["loss"]
        dataset_best = best.task_results[dataset] if best else None

        for i, (acc, loss) in enumerate(zip(accuracies, losses)):
            msg = f"      " +\
                  f"[Task {clr(f'{dataset:s}-{(i+1):03d}', attrs=['bold']):s}]"

            is_acc_better = dataset_best and dataset_best["acc"][i] < acc
            colors = ['white', 'on_magenta'] if is_acc_better else ['yellow']
            msg += f" Accuracy: {clr(f'{acc:5.2f}%', *colors):s}"

            is_loss_better = dataset_best and dataset_best["loss"][i] > loss
            colors = ['white', 'on_magenta'] if is_loss_better else ['yellow']
            msg += f" Loss: {clr(f'{loss:6.4f}', *colors):s}"

            print(msg)

        msg = f"   [Task {clr(f'{dataset:s}-ALL', attrs=['bold']):s}]"

        d_avg_acc = results.dataset_avg[dataset]["acc"]
        d_avg_loss = results.dataset_avg[dataset]["loss"]

        d_best_avg = best.dataset_avg[dataset] if best else None

        is_acc_better = d_best_avg and d_best_avg["acc"] < d_avg_acc
        colors = ['white', 'on_magenta'] if is_acc_better else ['yellow']
        msg += f" Accuracy: " +\
               f"{clr(f'{d_avg_acc:5.2f}%', *colors, attrs=['bold']):s}"

        is_loss_better = d_best_avg and d_best_avg["loss"] > d_avg_loss
        colors = ['white', 'on_magenta'] if is_loss_better else ['yellow']
        msg += f" Loss: " + \
               f"{clr(f'{d_avg_loss:6.4f}', *colors, attrs=['bold']):s}"

        print(msg)

    msg = f"[{clr('OVERALL', attrs=['bold']):s}]"

    g_acc, g_loss = results.global_avg["acc"], results.global_avg["loss"]

    is_acc_better = best and best.global_avg["acc"] < g_acc
    colors = ['white', 'on_magenta'] if is_acc_better else ['yellow']
    msg += f" Accuracy: {clr(f'{g_acc:5.2f}%', *colors, attrs=['bold']):s}"

    is_loss_better = best and best.global_avg["loss"] > g_loss
    colors = ['white', 'on_magenta'] if is_loss_better else ['yellow']
    msg += f" Loss: {clr(f'{g_loss:6.4f}', *colors, attrs=['bold']):s}"

    print(msg)


class Reporting:

    def __init__(self, args: Args):
        self._start_time = time.time()

    def trace_train(self, task_idx: int, epoch: int, seen: int, results: dict):
        pass

    def trace_eval(self, task_idx: int, epoch: int, seen: int, results: dict):
        pass



