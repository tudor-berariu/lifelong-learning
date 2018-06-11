from elasticsearch import Elasticsearch
import os
import pandas as pd
from typing import List, Dict, Callable, Any, Union, Tuple

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ES_SERVER = None

HOST = "localhost"
PORT = "9200"

eINDEX = "phd"
eDOC = "lifelong"


def mark_uploaded_name(file_path):
    dir_ = os.path.dirname(file_path)
    file_name_ = os.path.basename(file_path)
    mark_path = os.path.join(dir_, f".{file_name_}_uploaded")
    return mark_path


def init_elastic_client():
    """Init Elastic search client"""
    global ES_SERVER
    if ES_SERVER is None:
        ES_SERVER = Elasticsearch([{"host": HOST, 'port': PORT}])
    return ES_SERVER


def walk_complex_data(d: Dict,
                      lambda_keys: Callable[[Any], Any] = None,
                      lambda_value: Callable[[Any], Any] = None,
                      lambda_el: Callable[[Any], Any] = None):
    if isinstance(d, dict):
        response = []
        for k, v in d.items():
            if lambda_keys is not None:
                response.append(lambda_keys(k))
            if lambda_value is not None:
                response.append(lambda_value(v))
            response.extend(walk_complex_data(v))
        return response
    elif isinstance(d, list):
        response = []
        for v in d:
            response.extend(walk_complex_data(v))
        return response
    else:
        if lambda_el is not None:
            return [lambda_el(d)]
    return []


def walk_and_delete_key(d: Dict, keys: List[str]) -> None:
    """ Deletes all keys with name in list(keys) """
    if isinstance(d, dict):
        for k in keys:
            d.pop(k, None)
        for k, v in d.items():
            walk_and_delete_key(v, keys)
    elif isinstance(d, list):
        for v in d:
            walk_and_delete_key(v, keys)


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

    res = es.search(index=eINDEX, doc_type=eDOC, body=doc, scroll='1m')
    while len(res["hits"]["hits"]) > 0:
        hits = res["hits"]["hits"]

        # Update value:
        for hit in hits:
            len_datasets = len(hit["_source"]["args"]["tasks"]["datasets"])
            len_tasks = len(hit["_source"]["task_info"])
            es.update(index=eINDEX, doc_type=eDOC, id=hit["_id"], body={
                "doc": {
                    # "args": {"tasks": {"no_tasks": len_tasks, "no_datasets": len_datasets}}
                    "task_name": get_task_name(hit["_source"])
                }
            })

        scroll = res['_scroll_id']
        res = es.scroll(scroll_id=scroll, scroll='1m')


def clean_almost_all():
    """Example of script to clean all data on server except  """
    es = init_elastic_client()

    doc = {
        'size': 1000,
        'query': {
            'match_all': {}
        }
    }
    ignore_key = "GFs372MBm5wd3rDHv_cu"

    res = es.search(index=eINDEX, doc_type=eDOC, body=doc, scroll='1m')
    while len(res["hits"]["hits"]) > 0:
        hits = res["hits"]["hits"]

        # Update value:
        for hit in hits:
            if hit["_id"] != ignore_key:
                es.delete(index=eINDEX, doc_type=eDOC,  id=hit["_id"])

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

    res = es.search(index=eINDEX, doc_type=eDOC, body=query)
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
    res = es.search(index=eINDEX, doc_type=eDOC, body=doc, scroll='1m')
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


def get_hits_dsl_query(query: Dict, other: Dict = None) -> List[Dict]:
    """
        Can be used with queries of format: Query DSL | Elasticsearch
        Example:
                query = {
                    "match": {
                      "args.title": {
                        "query": "andrei",
                        "type": "phrase"
                      }
                    }
                  }
    """
    es = init_elastic_client()

    # parsing_exception', '[match] query does not support [type]')
    walk_and_delete_key(query, ["type"])

    doc = {
        'size': 1000,
        'query': query
    }
    if other is not None:
        doc.update(other)

    all_hits = []
    res = es.search(index=eINDEX, doc_type=eDOC, body=doc, scroll='1m')
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


def get_hits_dict_query(query: Dict) -> List[Dict]:
    """
        Can be used with queries of format:
        Dictionary with keys representing the field in database and value list of filter terms.
        Example argument:
        query = {
            "args.title" : ["andrei"]
            "args.tasks.datasets" : ["fashion", "mnist"]
        }


    """
    must_list = []
    dsl_query = {
         "bool" : {
            "must" : must_list
        }
    }

    for k, v in query.items():
        for term in v:
            must_list.append({"term": {k: term}})
    return get_hits_dsl_query(dsl_query)


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
            prefix + separator + str(k) if prefix else str(k): v
            for kk, vv in dd.items() for k, v in flatten_dict(vv, separator, str(kk)).items()
        }
    else:
        new_d = {prefix: dd}
    return new_d


def is_numeric(vv):
    try:
        a = float(vv)
    except ValueError:
        return False
    return True


def flatten_dict_keys(dd, separator='.', prefix=''):
    """ Transform complex data recursive to unique keys """
    if isinstance(dd, dict):
        all_k = set()
        for kk, vv in dd.items():
            k_name = "{int}" if is_numeric(kk) else kk
            all_k.update(flatten_dict_keys(vv, separator=separator,
                                           prefix=f"{prefix}{k_name}{separator}"))
        return all_k
    elif isinstance(dd, list):
        if len(dd) > 0:
            if isinstance(dd[0], dict):
                all_k = set()
                for vv in dd:
                    all_k.update(flatten_dict_keys(vv, separator=separator,
                                                   prefix=f"{prefix}{{int}}{separator}"))
                return all_k
            else:
                return set([f"{prefix}{str(type(dd))}"])
        return set([f"{prefix}{str(type(dd))}"])
    else:
        return set([f"{prefix}{str(type(dd))}"])


def update_fields_select_df(df: Union[pd.DataFrame, Any], new_fields: List[str],
                            update_file: str = None) -> Tuple(pd.DataFrame, List):

    if update_file is not None and os.path.isfile(update_file):
        df = pd.read_csv(update_file)

    all_fields = set(new_fields)
    if df is not None:
        all_fields.update(set(df["key"].values.tolist()))

    all_fields = sorted(list(set(all_fields)))
    new_df = pd.DataFrame(all_fields, columns=["key"])
    new_df["select"] = 0
    if df is not None:
        old_keys = df["key"].values.tolist()
        for idx in new_df.index:
            key = new_df.loc[idx, "key"]
            if key in old_keys:
                idx2 = old_keys.index(key)
                new_df.set_value(idx, "select", df.loc[idx2, "select"])

    new_df = new_df[["select", "key"]]

    if update_file is not None:
        new_df.to_csv(update_file)

    select_keys_col = new_df[new_df["select"] == 1]["key"].values.tolist()

    return new_df, select_keys_col


if __name__ == "__main__":

    get_hits_dsl_query(
        {
            "match": {
                "args.title": {
                    "query": "andrei",
                    "type": "phrase"
                }
            }
        }
    )

    get_hits_dict_query({"_id": ["F1s272MBm5wd3rDHf_es"]})