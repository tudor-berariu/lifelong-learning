from elasticsearch import Elasticsearch
import os
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
    from upload_to_elasticsearch import get_task_name

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
                    # "args": {"tasks": {"no_tasks": len_tasks, "no_datasets": len_datasets}}
                    "task_name": get_task_name(hit["_source"])
                }
            })

        scroll = res['_scroll_id']
        res = es.scroll(scroll_id=scroll, scroll='1m')


def clean_almost_all():
    """Example of script to use to update on server"""
    es = init_elastic_client()

    doc = {
        'size': 1000,
        'query': {
            'match_all': {}
        }
    }
    ignore_key = "OlsJ62MBm5wd3rDH8tVA"

    res = es.search(index="phd", doc_type='lifelong', body=doc, scroll='1m')
    while len(res["hits"]["hits"]) > 0:
        hits = res["hits"]["hits"]

        # Update value:
        for hit in hits:
            if hit["_id"] != ignore_key:
                es.delete(index="phd", doc_type='lifelong',  id=hit["_id"])

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


def get_all_hits():
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
            all_hits.append(hit)

        scroll = res['_scroll_id']
        res = es.scroll(scroll_id=scroll, scroll='1m')
    return all_hits


def get_data_as_dataframe():
    """Get all data in database to pandas dataframe"""
    all_hits = get_all_hits()
    flat_data = [flatten_dict(x) for x in all_hits]
    df = pd.DataFrame(flat_data)

    # df.to_csv("test.csv")
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
