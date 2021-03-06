from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as functional
from typing import List, Tuple, Union, Iterator
import numpy as np

from multi_task import TaskDataLoader, Batch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    val, avg, sum, count = None, None, None, None

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs: List[Tensor], targets: List[Tensor], topk=(1,)) -> List[Tuple[float, int]]:
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        res = []
        for output, target in zip(outputs, targets):
            res.append([])
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res[-1].append((correct_k.item(), batch_size))

        final_res = []
        for k in topk:
            k -= 1
            batch_size = np.sum([r[k][1] for r in res])
            correct = np.sum([r[k][0] for r in res])
            acc = correct / float(batch_size)
            final_res.append((acc * 100, correct))
        return final_res


def standard_train(train_loader: Union[TaskDataLoader, Iterator[Batch]], model: nn.Module,
                   optimizer: torch.optim.Optimizer, epoch: int,
                   batch_show_freq: int = -1,
                   max_batch: int = np.inf)-> Tuple[float, float, int]:

    losses = AverageMeter()
    acc = AverageMeter()
    correct_cnt = 0
    seen = 0.

    model.train()

    for batch_idx, (data, targets, head_idx) in enumerate(train_loader):
        if batch_idx > max_batch:
            break

        optimizer.zero_grad()
        outputs = model(data, head_idx=head_idx)

        loss = 0
        for out, target in zip(outputs, targets):
            loss += functional.cross_entropy(out, target)

        loss.backward()
        optimizer.step()

        (top1, correct), = accuracy(outputs, targets)
        correct_cnt += correct

        seen += data.size(0)
        acc.update(top1, data.size(0))
        losses.update(loss.item(), data.size(0))

        if batch_show_freq > 0 and (batch_idx + 1) % batch_show_freq == 0:
            print(f'\t\t[Train] [Epoch: {epoch:3}] [Batch: {batch_idx:5}]:\t '
                  f'[Loss] crt: {losses.val:3.4f}  avg: {losses.avg:3.4f}\t'
                  f'[Accuracy] crt: {acc.val:3.2f}  avg: {acc.avg:.2f}')

    return losses.avg, correct_cnt / float(seen), seen


def standard_validate(val_loader: TaskDataLoader, model: nn.Module, epoch: int,
                      report_freq: int = -1):
    losses = AverageMeter()
    acc = AverageMeter()
    correct_cnt = 0
    seen = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (data, targets, head_idx) in enumerate(val_loader):
            outputs = model(data, head_idx=head_idx)

            loss = 0
            for out, t in zip(outputs, targets):
                loss += functional.cross_entropy(out, t)

            (top1, correct), = accuracy(outputs, targets)
            correct_cnt += correct

            seen += data.size(0)
            acc.update(top1, data.size(0))
            losses.update(loss.item(), data.size(0))

            if (batch_idx + 1) % report_freq == 0 and report_freq > 0:
                print(f'\t\t[Eval] [Epoch: {epoch:3}] [Batch: {batch_idx:5}]:\t '
                      f'[Loss] crt: {losses.val:3.4f}  avg: {losses.avg:3.4f}\t'
                      f'[Accuracy] crt: {acc.val:3.2f}  avg: {acc.avg:.2f}')

        return losses.avg, correct_cnt / float(seen)


def get_ip() -> str:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


def redirect_std(out_filepath):
    import sys

    fsock = open(out_filepath, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = sys.stderr = fsock

    return fsock, old_stdout, old_stderr


def repair_std(out_filepath, fsock, old_stdout, old_stderr):
    import sys, shutil

    fsock.close()
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    print("=" * 79)
    with open(out_filepath, "r") as f:
        shutil.copyfileobj(f, sys.stdout)
    print("=" * 79)


def get_utc_time():
    import datetime
    import pytz
    return datetime.datetime.now(tz=pytz.utc)


def split_first_argument(ml: List[str]):
    # Parse first argument if it was given by <| xargs -0 -Ifoo>
    first_elem = ml[0]
    ml = ml[1:]
    first_elem = first_elem.split("\n")
    first_elem = [x for x in first_elem if len(x) > 0]
    ml.extend(first_elem)
    return ml
