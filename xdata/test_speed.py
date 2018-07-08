import torch
from torchvision.models import alexnet
import time
import numpy as np

if __name__ == "__main__":
    NO_TESTS = 100

    data = torch.rand(128, 3, 224, 224).cuda()

    model = alexnet()
    model.cuda()

    torch.cuda.synchronize()
    torch.cuda.synchronize()
    torch.backends.cudnn.benchmark = True

    a = time.perf_counter()
    model(data)
    torch.cuda.synchronize()  # wait for mm to finish
    b = time.perf_counter()


    tp = []
    for i in range(NO_TESTS):
        a = time.perf_counter()
        model(data)
        torch.cuda.synchronize()  # wait for mm to finish
        b = time.perf_counter()
        tp.append(b - a)

    tp.sort(reverse=True)
    print(tp)
    print(f"Mean top 25% {np.mean(tp[:int(len(tp) * 0.25)])}")
    print(f"Mean all:    {np.mean(tp)}")
