#!/home/tempuser/anaconda3/envs/andreiENV/bin python

import sys
from typing import List, Dict
from elasticsearch import Elasticsearch
import torch
from termcolor import colored as clr
import traceback
import shutil
import os
import json
import re

from utils.elasticsearch_utils import init_elastic_client, search_by_timestamp
from utils.util import repair_std, redirect_std

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


def upload_to_elastic(file_paths: List[str], force_update:bool = False):
    es: Elasticsearch = init_elastic_client()

    for file_path in file_paths:
        data: Dict = torch.load(file_path)
        data = fix_data(data)
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
    upload_to_elastic(sys.argv[1:])
