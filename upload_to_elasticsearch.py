#!/home/tempuser/anaconda3/envs/andreiENV/bin python
"""
    Upload either report.pkl files result of training or eData.pkl file that is a report.pkl file
        with additional processing (renaming variables, trimming...)
"""
import sys
from typing import List, Dict
from elasticsearch import Elasticsearch
import torch
from termcolor import colored as clr
import traceback
import argparse
import os
import json
import re
import glob
import numpy as np
import torch.multiprocessing as mp
import itertools

from utils.elasticsearch_utils import init_elastic_client, search_by_timestamp, eINDEX, eDOC
from utils.util import repair_std, redirect_std, split_first_argument
from utils.reporting import Reporting, BIG_DATA_KEYS

CHANGE = {
    " NaN": " 0",
    " Infinity": " 999999999",
    " -Infinity": " -999999999",
}

NON_SERIALIZABLE_KEYS = ["start_time", "end_time"]


def get_task_name(data):
    task_info = data["args"]["tasks"]
    task_name = f'#{"_".join(task_info["datasets"])}'
    task_name += f'_i{np.prod(task_info["in_size"])}'
    task_name += f'_rt{int(task_info["reset_targets"])}'
    task_name += f'_v{task_info["validation"]}'
    task_name += f'_s{max(task_info["split"], 1)}'
    task_name += f'_p{max(task_info["perms_no"], 1)}'
    task_name += f'_pt{int(task_info["permute_targets"])}'
    task_name += f'_c{int(task_info["common_head"])}'

    return task_name


def fix_data(data: Dict):
    # -- Remove non_mapping data such as (NaN inf)
    # Horrible hack
    non_serializable = dict()
    for k in NON_SERIALIZABLE_KEYS:
        if k in data:
            non_serializable[k] = data[k]

    serializable = data.copy()
    for k in NON_SERIALIZABLE_KEYS:
        serializable.pop(k, None)

    s = json.dumps(serializable)
    for k, v in CHANGE.items():
        s = re.sub(k, v, s)
    serializable = json.loads(s)

    data = non_serializable
    data.update(serializable)

    # -- Add redundant info but useful

    # add no_tasks to args.tasks.no_tasks
    # add no_datasets to args.tasks.no_datasets
    data["args"]["tasks"]["no_datasets"] = len(data["args"]["tasks"]["datasets"])
    data["args"]["tasks"]["no_tasks"] = len(data["task_info"])

    # Transform long dict to list
    best_eval = []
    last_eval = []
    for ix, v in sorted(data["best_eval"].items()):
        v["idx"] = ix
        best_eval.append(v)
    data["best_eval"] = best_eval
    for ix, v in sorted(data["last_eval"].items()):
        v["idx"] = ix
        last_eval.append(v)
    data["last_eval"] = last_eval

    data["best_eval_acc"] = [x["acc"]["value"] for x in best_eval]
    data["last_eval_acc"] = [x["acc"] for x in last_eval]

    # Extra columns ->

    data["task_name"] = get_task_name(data)

    # Fix values
    if data["args"]["train"]["stop_if_not_better"] is False:
        data["args"]["train"]["stop_if_not_better"] = 0

    return data


def upload_eData_to_elastic(args):
    p_idx, file_path, force_update = args

    print(f"[_{p_idx}_] Process eData")
    print(f"[_{p_idx}_] " + "=" * 79)

    es: Elasticsearch = init_elastic_client()
    indices = es.indices.get_alias("*")
    first_data = False
    if eINDEX not in indices:
        first_data = True

    if os.path.getsize(file_path) <= 0:
        print(f"[_{p_idx}_] " + f"[ERROR] File empty: {file_path}")
        return 333

    try:
        data = torch.load(file_path)
    except Exception as e:
        print(f"[_{p_idx}_] [ERROR] Can't open {file_path} err: {e}")
        return 334

    data = fix_data(data)
    match = None
    match_id = None

    if not first_data:
        res, found_items = search_by_timestamp(data["start_timestamp"])

        if found_items > 0:
            print(f"[_{p_idx}_] " +
                  f"[ERROR] Already found item in database (by timestamp): {file_path}")

            for h in res:
                # Check if same title
                title = h["_source"]["args"]["title"]
                run_id = h["_source"]["args"]["run_id"]
                if title == data["args"]["title"] and run_id == data["args"]["run_id"]:
                    match = h

            if match is None:
                print(f"[_{p_idx}_] " + f".... But not by name?!?!: {file_path}")
            elif not force_update:
                print(f"[_{p_idx}_] " + f"[ERROR] SKIP Duplicate")
                return 335
            else:
                match_id = match["_id"]
                print(f"[_{p_idx}_] " + f"Will update {file_path} -- match_id: {match_id}")

    out_filepath = file_path + "_out"
    fsock, old_stdout, old_stderr = redirect_std(out_filepath)
    res = None

    try:
        if force_update and match is not None:
            es.update(index=eINDEX, doc_type=eDOC, id=match_id, body={
                "doc": data
            })
        else:
            res = es.index(index=eINDEX,  doc_type=eDOC, body=data)
    except Exception as e:
        print(f"[_{p_idx}_] [ERROR] " + clr("COULD NOT PUSH TO SERVER!!!!!!!!!", "red"))
        print(f"[_{p_idx}_] " + "\nPLEASE do manual push :)")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=100, file=sys.stdout)
    if res:
        if res["result"] == "created":
            repair_std(out_filepath, fsock, old_stdout, old_stderr)

            os.remove(file_path)
            os.remove(out_filepath)

            print(f'[_{p_idx}_] INDEXED: {file_path} -- ID: {res["_id"]}')
        else:
            print(f'[_{p_idx}_] [ERROR] {file_path} -- result: {res["result"]}')
    else:
        print(f'[_{p_idx}_] [ERROR] {file_path} -- Res: {res}')

    print(f"[_{p_idx}_] " + "=" * 79)


def analyze_mapper_exection():
    import datetime, pytz

    data: Dict = torch.load("path")

    es = init_elastic_client()

    new_data = dict()
    for k, v in data.items():
        new_data[k] = v
        print(f"Added key: {k}")
        res = es.index(index='hw', doc_type='lifelong', body={"start": datetime.datetime.now(
            tz=pytz.utc), "test": 3})


def run_full_report_upload(args):
    pidx, file, args = args
    # Process reporting raw data
    print(f"[_{pidx}_] " + "Process reporing.pkl")
    print(f"[_{pidx}_] " + "=" * 79)

    Reporting.experiment_finished(file, ignore_keys=args.ignore_keys,
                                  local_efolder=args.local_efolder,
                                  mark_file_sent=args.mark_file_sent,
                                  force_reupload=args.force_reupload,
                                  force_update=args.force_update,
                                  push_to_server=args.push_to_server)
    print(f"[_{pidx}_] " + "=" * 79)
    print(f"[_{pidx}_] ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Upload data to Elasticserch server.\n'
                    '- All folders in list will be replaced by results.pkl file obtained '
                    'from a recursive search.\n'
                    '- All files if named results.pkl will be considered raw, else eData format')

    parser.add_argument(dest="paths", nargs='+', help='<Required> List of files / directories')
    parser.add_argument("--ignore-default", action="store_true", default=False,
                        help=f'Ignore default ignore keys ({BIG_DATA_KEYS})')
    parser.add_argument("--ignore-keys", nargs='+', help=f'Keys to ignore)')
    parser.add_argument("--local-efolder", default="results/tmp_efolder_data",
                        help=f'Folder where to move eData after processing)')
    parser.add_argument("--mark-file-sent",
                        type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True,
                        help=f'Mark results file as processed)')
    parser.add_argument("--force-reupload",
                        type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False,
                        help=f'Force re-upload of data if already marked as processed )')
    parser.add_argument("--force-update",
                        type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False,
                        help=f'Force update if data found in database )')
    parser.add_argument("--push-to-server",
                        type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True,
                        help=f'Push eData to server. )')
    parser.add_argument("-p", "--procs", type=int, action="store",
                        default=1, help=f'PROCS_NO')

    args = parser.parse_args()

    edata_file_paths = []
    results_file_paths = []

    for path in args.paths:
        if os.path.isfile(path):
            if os.path.basename(path) == "reporting.pkl":
                results_file_paths.append(os.path.abspath(path))
            else:
                edata_file_paths.append(os.path.abspath(path))
        else:
            results_files = glob.glob(f"{path}/**/reporting.pkl", recursive=True)
            results_files = [os.path.abspath(x) for x in results_files]
            results_file_paths.extend(results_files)

    # Process eData files first
    if len(edata_file_paths) > 0:
        p = mp.Pool(args.procs)
        p.map(upload_eData_to_elastic,
              zip(range(len(edata_file_paths)),
                  edata_file_paths,
                  itertools.repeat(args.force_update)))

    if args.ignore_default:
        ignore_keys = []
    else:
        ignore_keys = BIG_DATA_KEYS

    if args.ignore_keys is not None:
        ignore_keys.extend(args.ignore_keys)
    args.ignore_keys = ignore_keys

    if len(results_file_paths) > 0:
        p = mp.Pool(args.procs)
        p.map(run_full_report_upload,
              zip(range(len(results_file_paths)), results_file_paths, itertools.repeat(args)))
