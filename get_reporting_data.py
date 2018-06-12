import argparse
import pandas as p
import torch
import os
import glob
from typing import List, Dict, NamedTuple, Union
import re
import torch.multiprocessing as mp
import itertools

from utils.elasticsearch_utils import get_hits_dict_query, get_hits_dsl_query, \
    include_dict_complex_keys, exclude_dict_complex_keys
from utils.key_defines import KEY_SEPARATOR, KEY_SITERATOR


def read_report(file_path: str,
                include_keys: List[str] = list(), smart_group: Union[int, List[int]] = 0,
                exclude_keys: List[str] = list()) -> Dict:

    # TODO Bad fix for map pool not expanding arguments:
    if isinstance(file_path, tuple):
        file_path, include_keys, smart_group, exclude_keys = file_path

    report = {"__file_path": file_path}
    data = None

    # -- Try to read report
    try:
        data = torch.load(file_path)
    except Exception as e:
        print(f"[ERROR] Can't open {file_path} err: {e}")
        report["__error"] = str(e)
        return report

    if data is None:
        return report

    # -- Filter keys
    if len(exclude_keys) > 0:
        data = exclude_dict_complex_keys(data, exclude_keys=exclude_keys,
                                         separator=KEY_SEPARATOR, siterator=KEY_SITERATOR)
    if len(include_keys) > 0:
        data = include_dict_complex_keys(data, include_keys=include_keys, smart_group=smart_group,
                                         separator=KEY_SEPARATOR, siterator=KEY_SITERATOR)

    report.update(data)

    return report


def get_reports(base_path: str, e_ids: List[str] = list(), experiments: List[str] = list(),
                dir_regex_any: List[str] = list(), dir_regex_all: List[str] = list(),
                include_keys: List[str] = list(), smart_group: Union[int, List[int]] = 0,
                exclude_keys: List[str] = list(), no_proc: int = 1) -> List[Dict]:
    all_results = []
    match_file_filters = dict()

    # -- Get all paths of reporting.pkl files
    if len(e_ids) > 0:
        match_file_filters["e_idx"] = {}

        hits = get_hits_dsl_query({}, ids=e_ids, include_keys=["args.experiment", "args.out_dir"])

        not_found_ids = e_ids.copy()
        out_dirs = []
        for x in hits:
            out_dir = x["args"]["out_dir"]

            # Trim until first folder starting with digit
            if not out_dir[0].isdigit():
                match = re.finditer("/\d", out_dir)
                if match:
                    m = next(match)
                    out_dir = out_dir[m.span()[0]+1:]
                else:
                    print(f"[ERROR] out_dir not matching pattern <number>_experiment/ ({out_dir})")
                    continue

            not_found_ids.remove(x["_id"])
            out_dirs.append((x["_id"], out_dir))

        for id, out_dir in out_dirs:
            find_file = glob.glob(f"{base_path}/**/{out_dir}/**/reporting.pkl", recursive=True)
            all_results.extend(find_file)
            match_file_filters["e_idx"][id] = len(find_file)
    elif len(experiments) > 0:
        match_file_filters["experiments"] = {}

        for experiment in experiments:
            experiments_found = glob.glob(f"{base_path}/*_{experiment}/**/reporting.pkl",
                                         recursive=True)
            match_file_filters["experiments"][experiment] = len(experiments_found)
            all_results.extend(experiments_found)
    else:
        all_match = glob.glob(f"{base_path}/**/reporting.pkl", recursive=True)
        match_file_filters["all_match"] = len(all_match)
        all_results.extend(all_match)

    # -- Filter directories name
    if len(dir_regex_any) > 0:
        search_list = all_results
        all_results = []
        for path in search_list:
            accept = False
            for pattern in dir_regex_any:
                if len(re.findall(pattern, path)) > 0:
                    accept = True
                    break
            if accept:
                all_results.append(path)

    if len(dir_regex_all) > 0:
        search_list = all_results
        all_results = []
        for path in search_list:
            accept = True
            for pattern in dir_regex_all:
                if len(re.findall(pattern, path)) < 1:
                    accept = False
                    break
            if accept:
                all_results.append(path)

    all_results = [x for x in all_results if len(x) > 0]

    args = zip(
        all_results,
        itertools.repeat(include_keys), itertools.repeat(smart_group),
        itertools.repeat(exclude_keys),
    )
    pool = mp.Pool(no_proc)
    results = pool.map(read_report, args)
    pool.close()
    pool.join()

    return results


if __name__ == "__main__":
    import pprint
    parser = argparse.ArgumentParser(
        description='Gather reporting data. Priority argument is e-ids. If set all other filters '
                    'are ignored. Regex patterns do not exclude eachother')

    parser.add_argument("--args-path", help=f'Read arguments from file')
    parser.add_argument("--results-path", default="results/", help=f'Results folder base path')
    parser.add_argument("--save-path", default="results/get_reports_data.pkl",
                        help=f'Save path for results.')

    parser.add_argument("--no-procs", default=4, help=f'No processes')
    parser.add_argument("--experiments", nargs='+', default=[],
                        help='List of experiment names to filter')
    parser.add_argument("--e-ids", nargs='+', default=[],
                        help='Gather results with the following ids in the elastic search database')
    parser.add_argument("--dir-regex-any", nargs='+', default=[],
                        help='Must find at least one of this regex patterns in the folder name.')
    parser.add_argument("--dir-regex-all", nargs='+', default=[],
                        help='Must find at all this regex patterns in the folder name.')
    parser.add_argument("--include-keys", nargs='+', default=[], help='Data keys to include')
    parser.add_argument("--smart-group", nargs='+', default=[], help='Data keys to include')
    parser.add_argument("--exclude-keys", nargs='+', default=[], help='Data keys to exclude')

    args = parser.parse_args()

    args.delete_arg_file = False
    if args.args_path is not None:
        args = torch.load(args.args_path)

    if isinstance(args.smart_group, list):
        if len(args.smart_group) == 1:
            args.smart_group = args.smart_group[0]
        elif len(args.smart_group) == 0:
            args.smart_group = 1

    result = get_reports(args.results_path, e_ids=args.e_ids, experiments=args.experiments,
                         dir_regex_any=args.dir_regex_any, dir_regex_all=args.dir_regex_all,
                         include_keys=args.include_keys, smart_group=args.smart_group,
                         exclude_keys=args.exclude_keys,
                         no_proc=args.no_procs)

    torch.save(result, args.save_path)

