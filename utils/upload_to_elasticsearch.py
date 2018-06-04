#!/home/tempuser/anaconda3/envs/andreiENV/bin python

import sys
from typing import List
from elasticsearch import Elasticsearch
import torch
from termcolor import colored as clr
import traceback
import shutil
import os


def init_elastic_client():
    return Elasticsearch([{'host': 'localhost', 'port': 9200}])


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

        data = torch.load(file_path)

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


if __name__ == "__main__":
    upload_to_elastic(sys.argv[1:])
