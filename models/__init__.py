from .mlp import MLP
from .lenet import LeNet
from .kf_mlp import KFMLP
from .kf_lenet import KFLeNet

ALL_MODELS = {
    "mlp": MLP,
    "lenet": LeNet,
    "kf_mlp": KFMLP,
    "kf_lenet": KFLeNet
}

ALL_MODELS_BASE_TYPE = {
    "mlp": "mlp",
    "lenet": "lenet",
    "kf_mlp": "mlp",
    "kf_lenet": "lenet"
}


def get_model(name):
    # @name         : name of the model

    assert name in ALL_MODELS, "Model %s is not on defined." % name

    return ALL_MODELS[name]
