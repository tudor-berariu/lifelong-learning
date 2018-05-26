import os
from argparse import Namespace
import torch
import torch.optim as optim
from torch.optim import Optimizer
from typing import Type, Callable

from liftoff.config import value_of, namespace_to_dict
from liftoff.config import read_config

from models import get_model
from my_types import Args, Tasks, Model, LongVector, DatasetTasks
from train_indiviually import train_individually
from train_sequentially import train_sequentially
from train_simultaneously import train_simultaneously
from tasks import ORIGINAL_SIZE, get_tasks, permute, random_permute


def process_args(args: Args) -> Args:
    """Read command line arguments"""
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if isinstance(args.perms_no, int):
        args.perms_no = [args.perms_no for d in args.datasets]
    elif isinstance(args.perms_no, list) and len(args.perms_no) == 1:
        args.perms_no = args.perms_no * len(args.datasets)
    else:
        assert len(args.perms_no) == len(args.datasets), \
            "You must specify the number of permutations for each dataset."

    args.tasks_no = sum(args.perms_no)

    # Three combinations are supported for now:
    # - a single dataset with any number of permutations
    # - several datasets with the same number of permutations
    # - datasets with different # of permutations, but in this case
    #   the batch size must be a multiple of the total number of tasks

    d_no = len(args.datasets)
    batch_size, perms_no = args.train_batch_size, args.perms_no

    if args.mode != "sim":
        args.train_batch_size = [batch_size] * d_no
    elif d_no == 1:
        args.train_batch_size = [batch_size]
    elif len(set(perms_no)) == 1:
        args.train_batch_size = [batch_size // d_no] * d_no
    else:
        scale = batch_size / sum(perms_no)
        args.train_batch_size = [round(p_no * scale) for p_no in perms_no]

    if args.mode == "sim":
        print(f"Real batch_size will be {sum(args.train_batch_size):d}.")

    sizes = [ORIGINAL_SIZE[d] for d in args.datasets]
    in_sz = torch.Size([max([s[dim] for s in sizes]) for dim in range(3)])
    if args.up_size:
        assert len(args.up_size) == 3, "Please provide a valid volume size."
        in_sz = torch.Size([max(*p) for p in zip(in_sz, args.up_size)])
    args.in_size = in_sz
    print("Input size will be: ", in_sz)

    return args


def get_optimizer(model: Model, args: Args) -> Optimizer:
    kwargs = value_of(args, "optimizer_args", Namespace(lr=0.001))
    kwargs = namespace_to_dict(kwargs)
    return optim.__dict__[args.optimizer](model.parameters(), **kwargs)


def run(args: Args) -> None:
    print(torch.__version__)
    args = process_args(args)

    if hasattr(args, "_experiment_parameters"):
        for p_name, p_value in args._experiment_parameters.__dict__.items():
            print(p_name, p_value)

    # Model class, optimizer, tasks
    model_class = get_model(args)  # type: Model
    get_optim = lambda model: get_optimizer(model, args)  # type: Callable
    tasks = None  # type: Tasks

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    if args.mode == "sim":
        train_simultaneously(model_class, get_optim, tasks, args)
    elif args.mode == "seq":
        train_sequentially(model_class, get_optim, tasks, args)
    elif args.mode == "ind":
        train_individually(model_class, get_optim, tasks, args)


def main():

    # Reading args
    args = read_config()  # type: Args

    if not hasattr(args, "out_dir"):
        from time import time
        if not os.path.isdir('./results'):
            os.mkdir('./results')
        out_dir = f'./results/{str(int(time())):s}_{args.experiment:s}'
        os.mkdir(out_dir)
        args.out_dir = out_dir
    else:
        assert os.path.isdir(args.out_dir), "Given directory does not exist"

    if not hasattr(args, "run_id"):
        args.run_id = 0

    run(args)


if __name__ == "__main__":
    main()
