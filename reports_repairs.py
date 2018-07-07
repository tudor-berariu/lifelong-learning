from dateutil import parser as dateparser
from typing import Dict
import numpy as np
from copy import deepcopy
from liftoff.config import namespace_to_dict

from utils.reporting import Reporting, EVAL_METRICS_BASE, get_base_score


def repair_template(data: Dict, lprint = print, eprint = print):
    """
    Should apply fixes inplace on dictionary data

    :param data: Data dictionary
    :param lprint: function to use to log information
    :param eprint: function to use to log error information
    :return: Return code: 0 - Not necessary , 1 - Success, >=2 Error
    """
    return 0


def repair_last_eval(data: Dict, lprint=print, eprint=print):
    """ Repair Last eval (1) """

    if "_eval_trace" in data:
        eval_seen = sorted(data["_eval_trace"].keys())
        data["_last_eval"] = {k: v[-1] for k, v in data["_eval_trace"][eval_seen[-1]].items()}

        # Repair best eval
        best_eval = dict({task["idx"]: {"acc": {"value": -1, "seen": -1},
                                        "loss": {"value": np.inf, "seen": -1}}
                          for task in data["_task_info"]})
        last_eval = dict({task["idx"]: {"acc": -1, "seen": -1, "loss": np.inf}
                          for task in data["_task_info"]})

        for ix in eval_seen:
            for task_idx in data["_eval_trace"][ix].keys():
                for eval in data["_eval_trace"][ix][task_idx]:
                    Reporting.update_best(eval, last_eval, best_eval, task_idx, ix)

        data["_best_eval"] = best_eval
        return 1
    else:
        lprint(f"Does not have key: _eval_trace")

    return 0


def repair_task_train_tick(data: Dict, lprint=print, eprint=print):
    """ Repair Task Train tick """

    if data["_args"]["mode"] != "sim":

        lprint(f"Repair Task Train tick ...")

        all_seen = list(sorted(data["_eval_trace"].keys()))
        ev_trace = data['_eval_trace']
        epochs_per_task = data["_args"]["train"]["epochs_per_task"]
        ix = epochs_per_task - 1
        task_train_tick_idx = 0
        tasks = []
        while ix < len(all_seen):
            idx = all_seen[ix]
            task_eval_data = deepcopy(ev_trace[idx])
            for i, v in task_eval_data.items():
                task_eval_data[i] = v[-1]
                task_eval_data[i]["seen"] = idx
            data["_task_train_tick"][task_train_tick_idx]["task"] = task_eval_data
            task_train_tick_idx += 1
            ix += epochs_per_task

        return 1

    return 0


def repair_buggy_sim_metrics(data: Dict, lprint=print, eprint=print):
    """ Repair bugged Eval metrics calc for simultaneous mode """

    if data["_args"]["mode"] == "sim":
        lprint(f"Repair bugged Eval metrics calc for simultaneous mode ...")

        evm = data["_eval_metrics"]
        last_eval = data["_last_eval"]
        acc = []
        for ix, v in last_eval.items():
            acc.append(v["acc"])
        base = last_eval[0]["acc"]
        new = last_eval[max(last_eval.keys())]["acc"]
        all = np.mean(acc)
        evm.update({
            'score_new_raw': [new],
            'score_base_raw': [base],
            'score_all_raw': [[all]],
            'score_all_last_raw': [all],
            'score_new': new,
            'score_base': base,
            'score_all': all,
            'score_last': new})
        return 1

    return 0


def recalculate_eval_metrics(data: Dict, lprint=print, eprint=print):
    """ Recalculate metrics """

    lprint(f"Recalculate metrics ...")

    # Update missing keys
    eval_metrics = deepcopy(EVAL_METRICS_BASE)
    no_tasks = len(data["_task_info"])
    tasks_range = range(len(data["_task_info"]))
    base = get_base_score(data["_task_info"], data["_args"]["model"])

    # Update task base ind in task_info
    for x in data["_task_info"]:
        x["best_individual"] = base[x["idx"]]

    no_scale_base = {ix: 1. for ix in base.keys()}
    update_scores = Reporting.update_scores
    update_episodic_scores = Reporting.update_episodic_scores
    mode = data["_args"]["mode"]

    # Simulate Reporting.finished_training_task
    rep = (None, None, None, None)

    if mode != "seq":
        ev_trace = data['_eval_trace']
        seen = sorted(data['_eval_trace'].keys())
        task_eval_data = [[ev_trace[ix][tx][-1] for tx in tasks_range] for ix in seen]
        no_trained_tasks = [no_tasks] * len(seen)
        all_acc = [[ev_trace[ix][tx][-1]["acc"] for tx in tasks_range] for ix in seen]
    else:
        ttt = data["_task_train_tick"]
        ev_trace = data['_eval_trace']
        seen = [x["task"][0]["seen"] for x in ttt]
        task_eval_data = [x["task"] for x in ttt]
        no_trained_tasks = list(range(1, len(task_eval_data) + 1))
        all_acc = [x["global"]["acc"] for x in ttt]

    rep = zip(seen, task_eval_data, no_trained_tasks, all_acc)

    #  -- Calculate metrics
    for seen, task_eval_data, no_trained_tasks, all_acc in rep:
        if mode == "seq":
            eval_metrics["score_new_raw"].append(
                task_eval_data[no_trained_tasks - 1]["acc"])

            if 0 in task_eval_data:
                eval_metrics["score_base_raw"].append(task_eval_data[0]["acc"])
            else:
                eval_metrics["score_base_raw"].append(eval_metrics["score_base_raw"][-1])

            eval_metrics["score_all_raw"].append([])
            for key in sorted(task_eval_data):
                if key < no_trained_tasks:
                    eval_metrics["score_all_raw"][-1].append(task_eval_data[key]["acc"])
        else:
            # Fix for Simultanous mode (trained on all tasks)
            eval_metrics["score_new_raw"] = all_acc
            eval_metrics["score_base_raw"] = all_acc
            eval_metrics["score_all_raw"] = [all_acc]

        # Update base scores
        _ = update_scores(eval_metrics, no_scale_base, t="_")
        update_episodic_scores(eval_metrics, no_scale_base,
                               eval_metrics["_score_last"], all_acc, seen, t="_")
        _ = update_scores(eval_metrics, base)
        update_episodic_scores(eval_metrics, base, eval_metrics["score_last"], all_acc, seen)

    data["_eval_metrics"] = eval_metrics

    return 1


def missing_keys_4(data: Dict, lprint=print, eprint=print):
    """ Add keys: _max_eval_all_epoch, _max_seen_train, _max_seen_eval, _finished_experiment """

    if "_finished_experiment" not in data:
        lprint(f"Add keys _finished_experiment ...")

        max_eval = -1
        for k1, v1 in data["_eval_trace"].items():
            for k2, v2 in v1.items():
                max_eval += len(v2)
        max_train = -1
        for k1, v1 in data["_train_trace"].items():
            for k2, v2 in v1.items():
                max_train += len(v2)

        data["_max_eval_all_epoch"] = max_eval
        data["_max_train_all_epoch"] = max_train
        data["_max_seen_train"] = max_seen_train = max(data["_train_trace"].keys())
        data["_max_seen_eval"] = max_seen_eval = max(data["_eval_trace"].keys())

        # Check if finished or no
        no_tasks = len(data["_task_info"])
        epochs_per_task = data["_args"]["train"]["epochs_per_task"]
        should_train = no_tasks * epochs_per_task
        reached_max_train = should_train == max_train + 1
        same_seen = data["_max_seen_train"] == data["_max_seen_eval"]
        all_final_tasks_evaluated = len(data["_eval_trace"][max_seen_eval]) == no_tasks

        data["_finished_experiment"] = reached_max_train \
                                       and same_seen and all_final_tasks_evaluated

        return 1

    return 0


def repair_args_format(data: Dict, lprint=print, eprint=print):
    if not isinstance(data["_args"], dict):
        data["_args"] = namespace_to_dict(data["_args"])
        return 1
    return 0


REPAIRS = {
    0: (repair_last_eval, dateparser.parse("Jun 12 2018 12:00AM")),
    1: (repair_task_train_tick, dateparser.parse("Jun 12 2018 12:00AM")),
    2: (repair_buggy_sim_metrics, dateparser.parse("Jun 12 2018 12:00AM")),
    3: (recalculate_eval_metrics, dateparser.parse("Jun 12 2018 12:00AM")),
    4: (missing_keys_4, dateparser.parse("Jun 12 2018 12:00AM")),
    5: (repair_args_format, dateparser.parse("Jul 12 2018 12:00AM")),
}
