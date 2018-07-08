import torch
import torch.nn as nn
from torch.distributions import Categorical

BATCH_SIZE = 20
DATA_SIZE = 10
EPOCHS_NO = 100000


def main():
    # Scopul: să calculăm E[d_logp(y|x)/d_w] cu x~date, y~p(y|x)

    # Un model oarecare.
    # Pentru simplitate, același număr de unități pe fiecare strat.
    model = nn.Sequential(nn.Linear(DATA_SIZE, DATA_SIZE),
                          nn.ReLU(),
                          nn.Linear(DATA_SIZE, DATA_SIZE))
    log_softmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)

    # Datele: un singur batch :)
    data = torch.randn(BATCH_SIZE, DATA_SIZE)

    # ----------------------------------------------------------------------
    # Varianta I: eșantionăm repetat y din p(y|x) și facem media gradienților

    model.zero_grad()

    est_prob = torch.zeros(BATCH_SIZE, DATA_SIZE)

    for _epoch in range(EPOCHS_NO):
        logits = log_softmax(model(data))
        idx = Categorical(logits=logits).sample().unsqueeze(1).detach()
        logits.gather(1, idx).mean().backward()

        if _epoch % (10 ** 4 - 1) == 0:
            print(f">>> Epoch {_epoch + 1:6d}: " +
                  ", ".join([f"{name:s}={param.grad.div(_epoch + 1).abs().mean():f}"
                             for name, param in model.named_parameters()]))
        est_prob.scatter_add_(1, idx, torch.ones(idx.size()))

    print("Probabilities: ", est_prob / est_prob.sum(dim=1, keepdim=True))

    for name, param in model.named_parameters():
        print(f"\n{name:s}\n", param.grad.div(EPOCHS_NO))

    print("------------------------------------------------------------")

    # ----------------------------------------------------------------------
    # A doua metodă: backprop prin toate ieșirile ponderate cu p(y|x)

    model.zero_grad()

    logits = log_softmax(model(data))
    probs = logits.exp().detach()
    logits.mul(probs).sum().backward()  # logits.backward(probs)

    print("Probabilities: ", probs)

    for name, param in model.named_parameters():
        print(f"\n{name:s}\n", param.grad)

    print("------------------------------------------------------------")


if __name__ == "__main__":
    main()
