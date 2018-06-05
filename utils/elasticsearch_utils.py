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
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def mark_uploaded_name(file_path):
    dir_ = os.path.dirname(file_path)
    file_name_ = os.path.basename(file_path)
    mark_path = os.path.join(dir_, f".{file_name_}_uploaded")
    return mark_path


def init_elastic_client():
    """Init Elastic search client"""
    return Elasticsearch([{'host': 'localhost', 'port': 9200}])


def update_data():
    """Example of script to use to update on server"""
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


def search_by_timestamp(start_timestamp, es=None):
    query = {
        "query": {
            "match": {
                "start_timestamp": {
                    "query": start_timestamp,
                }
            }
        }
    }

    if es is None:
        es = init_elastic_client()

    res = es.search(index="phd", doc_type='lifelong', body=query)
    return res, len(res["hits"]["hits"])


def get_data_to_pandas():
    """Get all data in database to pandas dataframe"""
    es = init_elastic_client()

    doc = {
        'size': 1000,
        'query': {
            'match_all': {}
        }
    }

    all_hits = []
    res = es.search(index="phd", doc_type='lifelong', body=doc, scroll='1m')
    while len(res["hits"]["hits"]) > 0:
        hits = res["hits"]["hits"]

        # Update value:
        for hit in hits:
            source = hit.pop("_source")
            hit.update(source)
            hit = flatten_dict(hit)
            all_hits.append(hit)

        scroll = res['_scroll_id']
        res = es.scroll(scroll_id=scroll, scroll='1m')
    df = pd.DataFrame(all_hits)
    df.to_csv("test.csv")
    return df


def flatten_dict(dd, separator='.', prefix=''):
    if isinstance(dd, dict):
        new_d = {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items() for k, v in flatten_dict(vv, separator, kk).items()
        }
    else:
        new_d = {prefix: dd}
    return new_d
