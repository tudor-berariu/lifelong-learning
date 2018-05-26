import os
from argparse import Namespace
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from typing import Type, Callable, Tuple

from liftoff.config import value_of, namespace_to_dict
from liftoff.config import read_config

from models import get_model
from my_types import Args, Tasks, Model, LongVector, DatasetTasks
from train_individually import train_individually
from train_sequentially import train_sequentially
from train_simultaneously import train_simultaneously
from tasks import ORIGINAL_SIZE
from multi_task import MultiTask


def get_optimizer(args: Args) -> Tuple[Type, dict]:
    kwargs = value_of(args, "optimizer_args", Namespace(lr=0.001))
    kwargs = namespace_to_dict(kwargs)
    return optim.__dict__[args.optimizer], kwargs


def run(args: Args, multitask: MultiTask = None) -> None:
    print(torch.__version__)

    if hasattr(args, "_experiment_parameters"):
        for p_name, p_value in args._experiment_parameters.__dict__.items():
            print(p_name, p_value)

    # Model class, optimizer, tasks
    if multitask is None:
        multitask = MultiTask(args.tasks)  # type: MultiTask
    model_class = get_model(args.model.name)  # type: Type
    optimizer, optim_args = get_optimizer(args)

    def get_optim(model):
        optimizer(model, **optim_args)

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    if args.mode == "sim":
        train_simultaneously(model_class, get_optim, multitask, args)
    elif args.mode == "seq":
        train_sequentially(model_class, get_optim, multitask, args)
    elif args.mode == "ind":
        train_individually(model_class, get_optim, multitask, args)


def main(args):

    # Reading args
    if args is None:
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

    keep_alive = args.keep_alive

    if not keep_alive:
        run(args)
    else:
        # Keep multitask class

        multitask = MultiTask(args.tasks)  # type: MultiTask

        while True:
            args = read_config()  # type: Args
            os.system('clear')

            try:
                run(args, multitask=multitask)
            except KeyboardInterrupt:
                os.system('clear')
                print("Caught KeyboardInterrupt, worker terminated")
            finally:
                pass

            input("Press Enter to restart... ( ! Out directory will be reset ! )")
            os.rmdir(args.out_dir)
            os.mkdir(args.out_dir)


if __name__ == "__main__":
    main(None)
