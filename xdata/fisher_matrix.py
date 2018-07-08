import torch
import torch.nn as nn
from torch.distributions import Categorical

BATCH_SIZE = 1
DATA_SIZE = 7
EPOCHS_NO = 1000


def main():

    model = nn.Sequential(nn.Linear(DATA_SIZE, DATA_SIZE),
                          # nn.ReLU(),
                          nn.Linear(DATA_SIZE, DATA_SIZE),
                          nn.LogSoftmax(dim=1))
    data = torch.randn(BATCH_SIZE, DATA_SIZE)

    # First method: using samples from p(y|x)

    model.zero_grad()

    logits = model(data)
    probs = logits.exp().detach()

    estimated_probs = torch.zeros(BATCH_SIZE, DATA_SIZE)

    data_first = data.clone()
    for _ in range(1):
        for _epoch in range(EPOCHS_NO):
            logits = model(data_first)
            idx = Categorical(logits=logits).sample().unsqueeze(1).detach()
            logits.gather(1, idx).sum().backward()

            # Let's estimate p(y|x) based on observed samples
            estimated_probs.scatter_add_(1, idx, torch.ones(idx.size()))

        data_first = torch.randn(BATCH_SIZE, DATA_SIZE)

    print("Probabilities: ", estimated_probs / estimated_probs.sum(dim=1, keepdim=True))

    ratio = {}

    for name, param in model.named_parameters():
        print(f"\n{name:s}\n", param.grad.div(EPOCHS_NO))

        ratio[name] = param.grad.div(EPOCHS_NO).clone().detach()

    print("------------------------------------------------------------")
    # == other
    print("="*90)
    g = []
    sum_grad = torch.zeros_like(model[0].weight.grad)
    print(estimated_probs)
    for i in range(DATA_SIZE):
        r = 1
        # r = int(estimated_probs[0, i].item())
        for _ in range(r):
            model.zero_grad()
            logits = model(data)
            logits[0, i].backward()
            last_grad = model[0].weight.grad.detach().clone()
            g.append(last_grad)
            sum_grad.add_(last_grad * probs[0, i])
            # sum_grad.add_(last_grad)
    print(sum_grad)
    print("=" * 90)

    # ==

    # Second method: backward through all outputs weighted by p(y|x)

    model.zero_grad()

    logits = model(data)
    probs = logits.exp().detach()
    logits.mul(probs).sum().backward()  # logits.backward(probs)

    print("Probabilities: ", probs)

    for name, param in model.named_parameters():
        print(f"\n{name:s}\n", param.grad)
        ratio[name] /= param.grad

    print("\n\n Ratio:")
    for name, values in ratio.items():
        print(values)


if __name__ == "__main__":
    main()