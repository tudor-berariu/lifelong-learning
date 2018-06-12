import argparse
import pandas as p
import torch
import os
import glob
from typing import List, Dict

def get_reporting(base_path, e_idx: List[str] = [], experiments: List[str] = [],
                  regex_any: List[str] = [], regex_all: List[str] = [],
                  filter_keys: List[str] = []):

    if len(e_idx) > 0:
        results_files = glob.glob(f"{path}/**/reporting.pkl", recursive=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Gather reporting data. Priority argument is e-ids. If set all other filters '
                    'are ignored. Regex patterns do not exclude eachother')

    parser.add_argument("--results-path", default="results/", help=f'Results folder base path')
    parser.add_argument("--experiments", nargs='+', help='List of experiment names to filter')
    parser.add_argument("--e-ids", nargs='+', default=[],
                        help='Gather results with the following ids in the elastic search database')
    parser.add_argument("--regex-any", nargs='+', default=[],
                        help='Must find at least one of this regex patterns ')
    parser.add_argument("--regex-all", nargs='+', default=[],
                        help='Must find at all this regex patterns')
    parser.add_argument("--filter-keys", nargs='+', default=[], help='Data keys to filter')

    args = parser.parse_args()


    # results_files = [os.path.abspath(x) for x in results_files]
    # results_file_paths.extend(results_files)

    print(args.e_ids)
    # p = mp.Pool(args.procs)
    # p.map(run_full_report_upload,
    #       zip(range(len(results_file_paths)), results_file_paths, itertools.repeat(args)))
