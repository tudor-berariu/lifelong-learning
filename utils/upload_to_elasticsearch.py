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

CHANGE = {
    " NaN": " 0",
    " Infinity": " 999999999",
    " -Infinity": " -999999999",
}
NON_SERIALIZABLE_KEYS = ["start_time", "end_time"]


def init_elastic_client():
    return Elasticsearch([{'host': 'localhost', 'port': 9200}])


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


def upload_to_elastic(file_paths: List[str]):
    es: Elasticsearch = None

    for file_path in file_paths:
        out_filepath = file_path + "_out"
        fsock = open(out_filepath, 'w')
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = sys.stderr = fsock

        def repair_std():
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            print("=" * 79)
            with open(out_filepath, "r") as f:
                shutil.copyfileobj(f, sys.stdout)
            print("=" * 79)

        if not es:
            es = init_elastic_client()

        data: Dict = torch.load(file_path)

        data = fix_data(data)

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
            repair_std()
            fsock.close()
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
