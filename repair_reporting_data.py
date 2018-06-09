import argparse
import torch
from utils.reporting import Reporting
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Repair bugged versions.')

    parser.add_argument(dest="paths", nargs='+', help='<Required> List of reporting.pkl files')
    args = parser.parse_args()

    file_paths = args.paths

    for file_path in file_paths:
        print(f"Working on: {file_path}")

        data = torch.load(file_path)
        eval_seen = sorted(data["_eval_trace"].keys())

        # Repair Last eval
        data["_last_eval"] = {k: v[-1] for k, v in data["_eval_trace"][eval_seen[-1]].items()}

        # Repair best eval
        best_eval = dict({task["idx"]: {"acc": {"value": -1, "seen": -1},
                                           "loss": {"value": np.inf, "seen": -1}}
                                for task in data["_task_info"]})
        last_eval = dict({task["idx"]: {"acc": -1, "seen": -1, "loss":  np.inf}
                                for task in data["_task_info"]})

        for ix in eval_seen:
            for task_idx in data["_eval_trace"][ix].keys():
                for eval in data["_eval_trace"][ix][task_idx]:
                    Reporting.update_best(eval, last_eval, best_eval, task_idx, ix)

        data["_best_eval"] = best_eval

        torch.save(data, file_path)

        print(f"Done", "\n")





