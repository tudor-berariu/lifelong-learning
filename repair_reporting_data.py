import argparse
import torch
import datetime
import torch.multiprocessing as mp
import itertools
import os
import glob
from typing import Dict, NamedTuple, List, Tuple, Set
import time

from reports_repairs import REPAIRS

K_EXPERIMENT_TIMESTAMP = "_start_timestamp"
K_REPAIRS = "__repairs"
K_REPAIR_HISTORY = "__repair_history"

ORIGINAL_FORMAT = "_{}_original"
DATA_FILENAME = "reporting.pkl"

RepairHistory = NamedTuple(
    "RepairHistory",
    [("timestamp", float),
     ("repaired_keys", List[int])]
)


def get_winit(d: Dict, k: str, init_value=None):
    if k in d:
        return d[k]
    d[k] = init_value
    return d[k]


def get_function_name(function_ref) -> str:
    return str(function_ref).split()[1]


def repair_template(data: Dict, lprint = print, eprint = print):
    """
    Should apply fixes inplace on dictionary data

    :param data: Data dictionary
    :param lprint: function to use to log information
    :param eprint: function to use to log error information
    :return: Return code: 0 - Success, >0 - Errors
    """
    return 0


def get_original_path(path: str):
    file_path_basename = ORIGINAL_FORMAT.format(os.path.basename(path))
    return os.path.join(os.path.dirname(path), file_path_basename)


"""
    define repairs: 
    {
        KEY: (function, timestamp of repair implementation)
    }
"""

ERROR_CODES = {
    330 : "Can't open normal data {} err: {}",
    331 : "Can't open even original data {} err: {}",
    332 : "No data loaded!",
    333 : f"Experiment with no timestamp key ({K_EXPERIMENT_TIMESTAMP})"
}


def run_fix(_args: Tuple):
    p_idx, file_path, force, force_keys = _args
    original_file_path = get_original_path(file_path)

    # -- Define print functions
    def lprint(info):
        print(f"[_{p_idx}_] {info}")

    def eprint(info):
        print(f"[_{p_idx}_] [ERROR] {info}")

    lprint(f"Working on: {file_path}")

    # -- Load data dictionary
    data: Dict = None
    try:
        data = torch.load(file_path)
    except Exception as e:
        eprint(ERROR_CODES[330].format(file_path, e))
        lprint("Fallback to original file")

        # Try to load original data instead
        try:
            data = torch.load(original_file_path)
        except Exception as e:
            eprint(ERROR_CODES[331].format(original_file_path, e))
            return 331

    # Recheck if any data is loaded
    if data is None:
        eprint(ERROR_CODES[332])
        return 332

    # Identify experiment timestamp by
    if K_EXPERIMENT_TIMESTAMP not in data:
        eprint(ERROR_CODES[333])
        return 333

    tm = data[K_EXPERIMENT_TIMESTAMP]
    exp_date = datetime.datetime.utcfromtimestamp(int(tm))

    # Add keys of repairs
    repaired_keys: Set = get_winit(data, K_REPAIRS, set())
    repair_history: List = get_winit(data, K_REPAIR_HISTORY, list())

    crt_repair_k = []
    r_codes = []

    # -- Repair
    for repair_k, (function_ref, implementation_date) in REPAIRS.items():
        if (exp_date < implementation_date and repair_k not in repaired_keys) \
                or force or (repair_k in force_keys):

            cprint = lambda info: lprint(f"[KEY {repair_k:4}] {info}")

            # Repair
            cprint(f"* Try: {get_function_name(function_ref)} (Date: {implementation_date})")

            return_code = -1

            try:
                return_code = function_ref(data, lprint=cprint, eprint=eprint)
            except Exception as e:
                eprint(f"While repairing {repair_k}. error: {e}")

            # Done or not necessary
            if return_code == 1 or return_code == 0:
                # Did repair
                crt_repair_k.append(repair_k)
                cprint("Did repair")
            else:
                cprint(f"Did NOT do repair ({return_code})")

            r_codes.append((repair_k, return_code))

    repair_history.append(RepairHistory(time.time(), crt_repair_k))
    repaired_keys.update(crt_repair_k)

    if len(crt_repair_k) > 0:
        # If no original file, move crt data to original_file_path and save afterwards
        if not os.path.isfile(original_file_path):
            os.rename(file_path, original_file_path)

        torch.save(data, file_path)

    lprint(f"Done (Fix: {crt_repair_k})\n")
    lprint(f"(Repair_k, return_code) {r_codes}\n")

    return len(crt_repair_k), r_codes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Repair bugged versions.')

    parser.add_argument(dest="paths", nargs='+', help='<Required> List of reporting.pkl files')
    parser.add_argument("-p", "--procs", type=int, action="store",
                        default=1, help=f'PROCS_NO')
    parser.add_argument('-f', action="store_true", dest="force",
                        default=False, help=f'Force update all')
    parser.add_argument('-cf', dest="custom_force", nargs='+', type=int,
                        help='Custom keys to force update upon.', default=[])

    args = parser.parse_args()

    if os.path.isdir(args.paths[0]):
        args.paths = glob.glob(f"{args.paths[0]}/**/{DATA_FILENAME}", recursive=True)

    file_paths = args.paths

    cm = zip(
        range(len(file_paths)),
        file_paths,
        itertools.repeat(args.force),
        itertools.repeat(args.custom_force)
    )

    p = mp.Pool(args.procs)
    results = p.map(run_fix, cm)

    no_files = len(file_paths)
    repaired = 0
    with_errors = 0

    for file, (keys_repaired, repair_codes) in zip(file_paths, results):
        if keys_repaired > 0:
            repaired += 1

        for repair_k, return_code in repair_codes:
            if return_code not in [0, 1]:
                with_errors += 1
                print(f"File: {file}.\n\t Result: {repair_codes}")
                break

    print(f"\n\nScanned: {no_files}\nRepaired: {repaired}\nWith errors: {with_errors}")








