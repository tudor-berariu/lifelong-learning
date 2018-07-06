from elasticsearch import Elasticsearch
import os
import pandas as pd
from typing import List, Dict, Callable, Any, Union, Tuple
from copy import deepcopy
import numpy as np
import re

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ES_SERVER = None

HOST = "localhost"
PORT = "9200"

eINDEX = "phd"
eDOC = "lifelong"


def flatten_dict(dd, separator='.', prefix=''):
    if isinstance(dd, dict):
        new_d = {
            prefix + separator + str(k) if prefix else str(k): v
            for kk, vv in dd.items() for k, v in flatten_dict(vv, separator, str(kk)).items()
        }
        return new_d
    elif isinstance(dd, list):
        if len(dd) > 0:
            if isinstance(dd[0], dict):
                new_d = {
                    prefix + separator + str(k) if prefix else str(k): v
                    for kk, vv in enumerate(dd) for k, v in
                    flatten_dict(vv, separator, str(kk)).items()
                }
                return new_d

    new_d = {prefix: dd}
    return new_d


def is_numeric(vv):
    if vv is None:
        return False
    try:
        a = float(vv)
    except Exception as e:
        return False
    return True


def get_type_string(vv):
    s = str(type(float))
    return s.replace("<class '", "").replace("'>", "")


def flatten_dict_keys(dd, separator='.', prefix=''):
    """ Transform complex data recursive to unique keys """
    if isinstance(dd, dict):
        all_k = set()
        for kk, vv in dd.items():
            k_name = "[_]" if is_numeric(kk) else kk
            all_k.update(flatten_dict_keys(vv, separator=separator,
                                           prefix=f"{prefix}{k_name}{separator}"))
        return all_k
    elif isinstance(dd, list):
        if len(dd) > 0:
            if isinstance(dd[0], dict):
                all_k = set()
                for vv in dd:
                    all_k.update(flatten_dict_keys(vv, separator=separator,
                                                   prefix=f"{prefix}[_]{separator}"))
                return all_k
            else:
                return set([f"{prefix}[{get_type_string(dd)}]"])
        return set([f"{prefix}<{get_type_string(dd)}>"])
    else:
        return set([f"{prefix}<{get_type_string(dd)}>"])


def get_complex_key_recursive(dd: Dict, key: List[str], sep: str = ".", sit: str = "[_]") -> Dict:
    """ Get 1 complex key recursive """

    if len(key) < 1:
        return dd

    if re.match("\[.*\]", key[0]):
        if isinstance(dd, dict):
            res = {}
            for kk, vv in dd.items():
                res[kk] = get_complex_key_recursive(vv, key[1:], sep=sep, sit=sit)
            return res
        else:
            res = {}
            for kk, vv in enumerate(dd):
                res[kk] = get_complex_key_recursive(vv, key[1:], sep=sep, sit=sit)
            return res

    kk = key[0]

    while kk not in dd and not re.match("\[.*\]", key[0]):
        key = key[1:]
        if len(key) > 0:
            kk += sep + key[0]
        else:
            break

    if kk not in dd:
        return None

    return {kk: get_complex_key_recursive(dd[kk], key[1:], sep=sep, sit=sit)}


def rem_complex_key_recursive(dd: Dict, key: List[str], sep: str = ".", sit: str = "[_]"):
    """ Inplace Remove recursive complex key """

    if re.match("\[.*\]", key[0]):
        if isinstance(dd, dict):
            for kk, vv in dd.items():
                rem_complex_key_recursive(vv, key[1:], sep=sep, sit=sit)
        else:
            for kk, vv in enumerate(dd):
                rem_complex_key_recursive(vv, key[1:], sep=sep, sit=sit)

    kk = key[0]

    while kk not in dd and not re.match("\[.*\]", key[0]):
        key = key[1:]
        if len(key) > 0:
            kk += sep + key[0]
        else:
            break

    if kk not in dd:
        return

    if len(key) > 1:
        rem_complex_key_recursive(dd[kk], key[1:], sep=sep, sit=sit)
    else:
        dd.pop(kk)


def multi_index_df_to_dict(df, level=0) -> Dict:
    if level > 0:
        d = {}
        it = df.index.levels[0] if hasattr(df.index, "levels") else df.index
        for idx in it:
            d[idx] = multi_index_df_to_dict(df.loc[idx], level=level-1)
        return d
    elif isinstance(df, pd.DataFrame):
        d = {}
        for idx, df_select in df.groupby(level=[0]):
            d[idx] = df_select[0][0]
        return d
    else:
        return df[0]


def exclude_dict_complex_keys(data: Dict, exclude_keys: List[str],
                              separator: str =".", siterator: str ="[_]") -> Dict:
    """ Returns new dictionary without the specified complex keys """

    data = deepcopy(data)
    for key in exclude_keys:
        key = key.split(".")

        if key[-1].startswith("<") and key[-1].endswith(">"):
            key = key[:-1]

        rem_complex_key_recursive(data, key, sep=separator, sit=siterator)
    return data


def include_dict_complex_keys(data: Dict, include_keys: List[str],
                              smart_group: Union[int, List[int]] = 0,
                              separator: str =".", siterator: str ="[_]"):
    """ get only included keys from dictionary. """

    ret = {}

    smart_groups = smart_group
    if isinstance(smart_groups, list):
        assert len(smart_groups) == len(include_keys), "Len of smart_group must equal include_keys"
    else:
        smart_groups = [smart_group] * len(include_keys)

    for orig_key, smart_group in zip(include_keys, smart_groups):
        key = orig_key.split(".")

        if re.match("\[.*\]", key[-1]) or re.match("<.*>", key[-1]):
            key = key[:-1]

        key_data = get_complex_key_recursive(data, key, sep=separator, sit=siterator)

        if smart_group > 0:
            flat_data = flatten_dict(key_data)

            if not np.any(flat_data.values()):
                continue

            df = pd.DataFrame([x.split(separator) for x in flat_data.keys()])
            max_cl = df.columns.max()
            df["values"] = flat_data.values()

            df["common"] = ""
            common = []
            variable = []
            for i in range(max_cl+1):
                if len(df[i].unique()) == 1:
                    df["common"] += df[i] + separator
                    common.append(i)
                else:
                    variable.append(i)
            df = df.drop(common, axis=1)

            for col in df.columns:
                if is_numeric(df.loc[0, col]) and col != "values":
                    df.loc[:, col] = df[col].apply(lambda x: int(float(x)))

            # Merge common columns
            index_col = [df["common"].values] + [df[x].values for x in variable]
            index = pd.MultiIndex.from_arrays(index_col, names=range(len(index_col)))
            df_index = pd.DataFrame(df["values"].values, index=index)

            # Only if smart group > 1 drop indexes
            group = 1
            index_level = len(index.levels) - 2

            while group < smart_group and index_level >= 0:
                index_tuple = []
                values = []
                for date, new_df in df_index.groupby(level=index_level):
                    values.append(new_df[0].values)
                    index_tuple.append(new_df.index.values[0][:-1])
                index = pd.MultiIndex.from_tuples(index_tuple)
                df_index = pd.DataFrame([0] * len(values), index=index)
                df_index.loc[:, 0] = pd.Series(values).values
                group += 1
                index_level -= 1

            l = 0 if len(df_index.index.levels) == 1 else len(df_index.index.levels)
            key_data = multi_index_df_to_dict(df_index, l)

        ret[orig_key] = key_data

    return ret


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

    # doc = {
    #     'size': 1000,
    #     "query":  {"bool": { "must": {
    #         "term": {
    #             "args.experiment": "test_depth",
    #         }
    #     }}}
    # }

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
        'size': 1000,
        "query":  {"bool": { "must": {
            "term": {
                "start_timestamp": start_timestamp,
            }
        }}}
    }

    if es is None:
        es = init_elastic_client()

    res = es.search(index=eINDEX, doc_type=eDOC, body=query, scroll='1m')

    # Filter for double precision:
    hits = []
    while len(res["hits"]["hits"]) > 0:
        # Update value:
        hits.extend([x for x in res["hits"]["hits"] if x["_source"]["start_timestamp"] ==
                           start_timestamp])

        scroll = res['_scroll_id']
        res = es.scroll(scroll_id=scroll, scroll='1m')

    return hits, len(hits)


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


def format_complex_key_for_elastic(complex_key: str, sep: str = ".") -> str:
    keys = complex_key.split(sep)
    if re.match("\[.*\]", keys[-1]) or re.match("<.*>", keys[-1]):
        keys = keys[:-1]

    keys = sep.join(keys)

    keys = re.sub("\[[^\]]*\].", "*", keys)
    return keys


def get_hits_dsl_query(query: Dict, other: Dict = None, ids: List[str] = list(),
                       include_keys: List[str] = list(), exclude_keys: List[str] = list(),
                       df_format=False) -> List[Dict]:
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
        "_source": {},
        'size': 1000,
        'query': query
    }

    if len(ids) > 0:
        query["ids"] = {"values": ids}
    if len(include_keys) > 0:
        include_keys = [format_complex_key_for_elastic(x) for x in include_keys]
        doc["_source"]["includes"] = include_keys
    if len(exclude_keys) > 0:
        exclude_keys = [format_complex_key_for_elastic(x) for x in exclude_keys]
        doc["_source"]["excludes"] = exclude_keys

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

    if df_format:
        flat_data = [flatten_dict(x) for x in all_hits]
        all_hits = pd.DataFrame(flat_data)

    return all_hits


def get_hits_dict_query(query: Dict, ids: List[str] = list(),
                        include_keys: List[str] = list(), exclude_keys: List[str] = list(),
                        df_format=False) -> List[Dict]:
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
         "bool": {
            "must": must_list
         }
    }

    for k, v in query.items():
        for term in v:
            must_list.append({"term": {k: term}})
    res = get_hits_dsl_query(dsl_query, ids=ids,
                             include_keys=include_keys, exclude_keys=exclude_keys,
                             df_format=df_format)
    return res


def table_data_by_depth(data: Dict, unpack_list: bool = False) -> Tuple[List[Dict], int]:
    """
        Backtrack to be possible to handle large data.
    """
    original = deepcopy(data)

    table_data = []
    stack = []
    crt_row = {}
    max_depth = 0
    crt_dict = original
    while len(original) > 0:
        while len(crt_dict.keys()) > 0:
            kk = next(iter(crt_dict))
            vv = crt_dict[kk]
            if isinstance(vv, dict):
                stack.append(crt_dict)
                crt_row[len(crt_row.keys())] = kk
                crt_dict = vv
            elif (isinstance(vv, list) or isinstance(vv, np.ndarray)) and unpack_list:
                crt_dict[kk] = {k: v for k, v in enumerate(vv)}
            else:
                new_row = deepcopy(crt_row)
                new_row[len(new_row.keys())] = kk
                new_row[len(new_row.keys())] = vv
                max_depth = max(max_depth, len(new_row.keys()))
                table_data.append(new_row)
                crt_dict.pop(kk, None)

        if len(stack) > 0:
            crt_dict = stack.pop()
            last_k = crt_row.pop(len(stack))
            crt_dict.pop(last_k)

    return table_data, max_depth


def get_data_as_dataframe():
    """Get all data in database to pandas dataframe"""
    all_hits = get_all_hits()
    flat_data = [flatten_dict(x) for x in all_hits]
    df = pd.DataFrame(flat_data)

    # df.to_csv("test.csv")
    return df


def update_fields_select_df(df: Union[pd.DataFrame, Any], new_fields: List[str],
                            update_file: str = None) -> Tuple[pd.DataFrame, List[str], List[int]]:

    if update_file is not None and os.path.isfile(update_file):
        df = pd.read_csv(update_file)

    all_fields = set(new_fields)
    if df is not None:
        all_fields.update(set(df["key"].values.tolist()))

    all_fields = sorted(list(set(all_fields)))
    new_df = pd.DataFrame(all_fields, columns=["key"])
    new_df["select"] = 0
    new_df["smart_group"] = 1
    if df is not None:
        old_keys = df["key"].values.tolist()
        for idx in new_df.index:
            key = new_df.loc[idx, "key"]
            if key in old_keys:
                idx2 = old_keys.index(key)
                new_df.set_value(idx, "select", df.loc[idx2, "select"])

    new_df = new_df[["select", "smart_group", "key"]]

    if update_file is not None:
        new_df.to_csv(update_file)

    selected = new_df[new_df["select"] == 1]
    select_keys_col = selected["key"].values.tolist()
    select_smart_group_col = selected["smart_group"].values.tolist()

    return new_df, select_keys_col, select_smart_group_col


def get_variable_key_names(complex_key: str, base_name: str, sep:str = ".",
                           siterator: str = "[_]") -> List[str]:

    empty = re.findall("\[([^\]]*)\]", siterator)[0]
    match = re.findall("\[([^\]]*)\]", complex_key)
    if not base_name.endswith(sep):
        base_name += sep
    return [x if x != empty else f"{base_name}{ix}" for ix, x in enumerate(match)]


def convert_report_to_df(report: Dict, sep=".", siterator: str = "[_]", unpack_list: bool=False):
    """
        Transform report format to dataframe.
        single key information will be distributed to all classes
        dictionaries will be merged on comm
    """
    infos = [x["info"] for x in report]
    datas = [x["data"] for x in report]

    reports_df = []

    for ix, data in enumerate(datas):

        infos[ix]["complex_key"] = []
        full_df = None

        for k, v in data.items():
            if v is not None:
                infos[ix]["complex_key"].append(k)

                assert len(v.keys()) == 1, "First key should be the common part of complex_key"
                common_key = next(iter(v))
                variable_data = v[common_key]

                if isinstance(variable_data, dict):
                    # new_d = flatten_dict(variable_data)
                    variable_data, _ = table_data_by_depth(variable_data, unpack_list=unpack_list)
                    key_df = pd.DataFrame(variable_data)
                    if len(key_df.columns) > 1:
                        # Check to see if named variable columns
                        col_names = get_variable_key_names(k, common_key,
                                                           sep=sep, siterator=siterator)
                        col_names = col_names[:len(key_df.columns)]
                        key_df.columns = col_names
                    else:
                        key_df.columns = [common_key]
                else:
                    key_df = pd.DataFrame([0], columns=[common_key], dtype = np.object)
                    key_df.loc[0, common_key] = variable_data

                key_df["_match_"] = 0

                if full_df is None:
                    full_df = key_df
                else:
                    common_col = list(set(key_df.columns) & set(full_df.columns))
                    full_df = full_df.merge(key_df, how='left', on=common_col)

        if full_df is not None:
            full_df = full_df.drop(["_match_"], axis=1)

        full_df["reporting_idx"] = ix

        reports_df.append(full_df)

    if len(reports_df) > 0:
        reports_df = pd.concat(reports_df, ignore_index=True)
    else:
        reports_df = None

    return reports_df, infos


def get_server_reports(e_ids: List[str] = list(), experiments: List[str] = list(),
                       dir_regex_any: List[str] = list(), dir_regex_all: List[str] = list(),
                       include_keys: List[str] = list(), smart_group: Union[int, List[int]] = 0,
                       exclude_keys: List[str] = list(), no_proc: int = 1,
                       df_format: bool = False):

    from utils.key_defines import REMOTE_HOST, SERVER_RESULTS, SERVER_eFOLDER, \
        SERVER_GET_REPORT_SCRIPT, SERVER_PYTHON
    import subprocess
    import torch
    from argparse import Namespace
    import time
    from utils.pid_wait import wait_pid

    full_report = {}
    df_return = None

    report_name = f"report_{int(time.time())}.pkl"
    server_path = os.path.join(SERVER_eFOLDER, report_name)

    args = Namespace()
    args.results_path = SERVER_RESULTS
    args.e_ids = e_ids
    args.experiments = experiments
    args.dir_regex_any = dir_regex_any
    args.dir_regex_all = dir_regex_all
    args.include_keys = include_keys
    args.smart_group = smart_group
    args.exclude_keys = exclude_keys
    args.no_procs = no_proc

    args.save_path = server_path

    local_args_file = "results/args.pkl"
    local_report = "results/full_report.pkl"

    torch.save(args, local_args_file)

    # -- Send arguments by file
    print("Send arguments ...")
    p = subprocess.Popen(["scp", local_args_file, f"{REMOTE_HOST}:{server_path}"])
    sts = wait_pid(p.pid, timeout=600)

    # -- Run script
    print("Run remote script ...")
    p = subprocess.Popen( f"ssh {REMOTE_HOST} {SERVER_PYTHON} {SERVER_GET_REPORT_SCRIPT} "
                          f"--args-path {server_path}", shell=True)
    sts = wait_pid(p.pid, timeout=120)

    # -- Receive response back
    print("Receive response data ...")
    p = subprocess.Popen(["scp", f"{REMOTE_HOST}:{server_path}", local_report],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sts = wait_pid(p.pid, timeout=120)

    if os.path.isfile(local_report):
        full_report = torch.load(local_report)

        if df_format:
            print("Convert to dataframe format ...")
            df_return = convert_report_to_df(full_report)

    return full_report, df_return


if __name__ == "__main__":
    pass
    # get_hits_dsl_query(
    #     {
    #         "match": {
    #             "args.title": {
    #                 "query": "andrei",
    #                 "type": "phrase"
    #             }
    #         }
    #     }
    # )
    #
    # get_hits_dict_query({"_id": ["F1s272MBm5wd3rDHf_es"]})

    # d = get_server_reports(["F1s272MBm5wd3rDHf_es"])
    # print(d)