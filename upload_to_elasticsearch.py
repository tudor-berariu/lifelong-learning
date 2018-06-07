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

from utils.elasticsearch_utils import init_elastic_client, search_by_timestamp
from utils.util import repair_std, redirect_std
from utils.reporting import Reporting, BIG_DATA_KEYS

CHANGE = {
    " NaN": " 0",
    " Infinity": " 999999999",
    " -Infinity": " -999999999",
}

NON_SERIALIZABLE_KEYS = ["start_time", "end_time"]


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

    # add no_tasks to args.tasks.no_tasks
    # add no_datasets to args.tasks.no_datasets
    data["args"]["tasks"]["no_datasets"] = len(data["args"]["tasks"]["datasets"])
    data["args"]["tasks"]["no_tasks"] = len(data["task_info"])

    return data


def upload_eData_to_elastic(file_paths: List[str], force_update: bool = False):
    es: Elasticsearch = init_elastic_client()
    indices = es.indices.get_alias("*")
    first_data = False
    if "phd" not in indices:
        first_data = True

    for file_path in file_paths:
        data: Dict = torch.load(file_path)
        data = fix_data(data)

        if not first_data:
            _, found_items = search_by_timestamp(data["start_timestamp"])

            if not force_update and found_items > 0:
                print(f"[ERROR] Already found item in database: {file_path}")
                continue

        out_filepath = file_path + "_out"
        fsock, old_stdout, old_stderr = redirect_std(out_filepath)

        try:
            res = es.index(index='phd',  doc_type='lifelong', body=data)
        except Exception as e:
            print(clr("COULD NOT PUSH TO SERVER!!!!!!!!!", "red"))
            print(clr("COULD NOT PUSH TO SERVER!!!!!!!!!", "red"))
            print(clr("COULD NOT PUSH TO SERVER!!!!!!!!!", "red"))
            print("\nPLEASE do manual push :)")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=100, file=sys.stdout)

        if res["result"] == "created":
            repair_std(out_filepath, fsock, old_stdout, old_stderr)

            os.remove(file_path)
            os.remove(out_filepath)

        print(f"INDEXED: {file_path}")


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
    print("Process eData")
    print("=" * 79)
    upload_eData_to_elastic(edata_file_paths, force_update=args.force_update)
    print("=" * 79)

    # Process reporting raw data
    print("Process reporing.pkl")
    print("=" * 79)

    if args.ignore_default:
        ignore_keys = []
    else:
        ignore_keys = BIG_DATA_KEYS

    if args.ignore_keys is not None:
        ignore_keys.extend(args.ignore_keys)

    for file in results_file_paths:
        Reporting.experiment_finished(file, ignore_keys=ignore_keys,
                                      local_efolder=args.local_efolder,
                                      mark_file_sent=args.mark_file_sent,
                                      force_reupload=args.force_reupload,
                                      force_update=args.force_update,
                                      push_to_server=args.push_to_server)

    print("=" * 79)

