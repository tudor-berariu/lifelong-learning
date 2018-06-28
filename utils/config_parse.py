from argparse import ArgumentParser
import torch
import os.path
import re
import numpy as np

"""
Match pattern:
    ${<PATTERN>, <parameters_comma_seperated>....}

E.g.
    seed: ${list, 1,2,3}
    seed: ${eval, np.random.rand(10)}
"""


def list_of_param(args_string):
    a = args_string.strip().split(",")
    return a


PATTERN = {
    "list": list_of_param,
    "eval": eval
}


def is_pattern_type(parameter: str):
    mtches = re.findall("\$\{(.*?)\}", parameter)
    if len(mtches) == 1:
        return True
    return False


def eval_pattern(pattern: str):
    if is_pattern_type(pattern):
        mtches = re.findall("\$\{(.*?)\}", pattern)
        pattern = mtches[0]

        method = re.findall("[^,]*", pattern)[0]
        args_str = pattern.replace(f"{method},", "")
        return PATTERN[method](args_str)

    return pattern
