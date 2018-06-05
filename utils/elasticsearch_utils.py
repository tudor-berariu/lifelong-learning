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


def init_elastic_client():
    return Elasticsearch([{'host': 'localhost', 'port': 9200}])


def iterate_date():
    es = init_elastic_client()

    doc = {
        'size': 1000,
        'query': {
            'match_all': {}
        }
    }

    res = es.search(index="phd", doc_type='lifelong', body=doc, scroll='1m')
    while len(res["hits"]["hits"]) > 0:
        hits = res["hits"]["hits"]

        # Update value:
        for hit in hits:
            len_datasets = len(hit["_source"]["args"]["tasks"]["datasets"])
            len_tasks = len(hit["_source"]["task_info"])
            es.update(index="phd", doc_type='lifelong', id=hit["_id"], body={
                "doc": {
                    "args": {"tasks": {"no_tasks": len_tasks, "no_datasets": len_datasets}}
                }
            })

        scroll = res['_scroll_id']
        res = es.scroll(scroll_id=scroll, scroll='1m')
