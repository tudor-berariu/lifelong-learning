import argparse
import torch
from copy import deepcopy
from dateutil import parser as dateparser
import datetime
import torch.multiprocessing as mp
import itertools
from liftoff.config import namespace_to_dict
import os
import glob

from utils.reporting import Reporting, EVAL_METRICS_BASE, get_base_score, EPISODIC_METRICS
from utils.util import split_first_argument
import numpy as np


def run_fix(args):
    p_idx, file_path, force = args
    print(f"[_{p_idx}_] Working on: {file_path}")

    try:
        data = torch.load(file_path)
    except Exception as e:
        print(f"[_{p_idx}_] [ERROR] Can't open {file_path} err: {e}")
        return 1

    fix = False
    if "_start_timestamp" not in data:
        tm = data["_start_time"]
    else:
        tm = data["_start_timestamp"]

    start_time = datetime.datetime.utcfromtimestamp(int(tm))

    if not isinstance(data["_args"], dict):
        data["_args"] = namespace_to_dict(data["_args"])

    # ==============================================================================================
    # -- Repair Last eval
    if start_time < dateparser.parse("Jun 12 2018 12:00AM") or force:
        print(f"[_{p_idx}_] Repair Last eval ...")

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
            fix = True
        else:
            print(f"[_{p_idx}_] Does not have key: _eval_trace")

    # ==============================================================================================
    # -- Repair Task Train tick
    if start_time < dateparser.parse("Jun 12 2018 12:00AM") or force:
        if data["_args"]["mode"] != "sim":

            print(f"[_{p_idx}_] Repair Task Train tick ...")

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

    # ==============================================================================================
    # -- Repair bugged Eval metrics calc for simultaneous mode
    if start_time < dateparser.parse("Jun 12 2018 12:00AM") or force:
        print(f"[_{p_idx}_] Repair bugged Eval metrics calc for simultaneous mode ...")

        if data["_args"]["mode"] == "sim":
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
            fix = True

    # ==============================================================================================
    # -- Recalculate metrics

    if start_time < dateparser.parse("Jun 12 2018 12:00AM") or force:
        print(f"[_{p_idx}_] Recalculate metrics ...")

        # Update missing keys
        eval_metrics = deepcopy(EVAL_METRICS_BASE)
        no_tasks = len(data["_task_info"])
        tasks_range = range(len(data["_task_info"]))
        base = get_base_score(data["_task_info"], data["_args"]["model"]["name"])
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

        fix = True

    if fix:
        torch.save(data, file_path)

    print(f"[_{p_idx}_] Done (Fix: {fix})", "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Repair bugged versions.')

    parser.add_argument(dest="paths", nargs='+', help='<Required> List of reporting.pkl files')
    parser.add_argument("-p", "--procs", type=int, action="store",
                        default=1, help=f'PROCS_NO')
    parser.add_argument('-f', action="store_true", dest="force",
                        default=False, help=f'Force update all')

    args = parser.parse_args()

    if os.path.isdir(args.paths[0]):
        args.paths = glob.glob(f"{args.paths[0]}/**/reporting.pkl", recursive=True)

    file_paths = args.paths

    cm = zip(
        range(len(file_paths)),
        file_paths,
        itertools.repeat(args.force)
    )

    p = mp.Pool(args.procs)
    p.map(run_fix, cm)





