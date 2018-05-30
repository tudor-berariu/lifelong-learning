import sys
import os
from argparse import Namespace
import torch
import torch.optim as optim
from typing import Type, Tuple
from shutil import rmtree
import importlib


from liftoff.config import value_of, namespace_to_dict
from liftoff.config import read_config

from models import get_model
from my_types import Args
from multi_task import MultiTask


def get_optimizer(args: Args) -> Tuple[Type, dict]:
    kwargs = value_of(args, "optimizer_args", Namespace(lr=0.001))
    kwargs = namespace_to_dict(kwargs)
    return optim.__dict__[args.optimizer], kwargs


def run(args: Args, multitask: MultiTask = None) -> None:
    print(torch.__version__)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.device = device = "cuda:0" if args.cuda else "cpu"
    for key, value in args.__dict__.items():
        if isinstance(value, Namespace):
            setattr(value, "device", device)

    if hasattr(args, "_experiment_parameters"):
        for p_name, p_value in args._experiment_parameters.__dict__.items():
            print(p_name, p_value)

    # Model class, optimizer, tasks
    if multitask is None:
        multitask = MultiTask(args)  # type: MultiTask

    model_class = get_model(args.model.name)  # type: Type
    optimizer, optim_args = get_optimizer(args.train)

    def get_optim(model) -> optim.Optimizer:
        return optimizer(model, **optim_args)

    def init_model(*args_, **kwargs_):
        model = model_class(*args_, **kwargs_)
        model = model.to(args.device)
        return model

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    if args.mode == "sim":
        import train_simultaneously
        train_simultaneously.train_simultaneously(init_model, get_optim, multitask, args)
    elif args.mode == "ind":
        import train_individually
        train_individually.train_individually(init_model, get_optim, multitask, args)
    elif args.mode == "seq":
        from algorithms import get_algorithm
        train_sequentially = get_algorithm(args.lifelong.mode)
        train_sequentially(model_class, get_optim, multitask, args)


def main(args: Args):

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
        import traceback

        # Keep multitask class & redo experiment with reload of config_file & modules

        multitask = MultiTask(args)  # type: MultiTask

        while True:
            # TODO Read again the config
            # args = read_config()  # type: Args
            os.system('clear')

            try:
                run(args, multitask=multitask)
            except KeyboardInterrupt:
                print("\nKeyboard interupted\n")
            except Exception as e:
                os.system('clear')
                print("Caught Error, worker terminated\n")
                print(e + "\n")
                traceback.print_exc()
                print()
            finally:
                pass

            input("Press Enter to restart... ( ! Out directory will be reset ! )")

            # Reset necessary (remove out directory)
            rmtree(args.out_dir)
            #
            # # Reload modules (not recommended)
            # importlib.reload(train_individually)
            # importlib.reload(train_simultaneously)
            # importlib.reload(train_sequentially)


if __name__ == "__main__":
    main(None)
