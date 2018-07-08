from .mlp import MLP
from .lenet import LeNet
from .kf_mlp import KFMLP
from .kf_lenet import KFLeNet
from .mlp_units_mask import MaskedMLP

ALL_MODELS = {
    "mlp": MLP,
    "lenet": LeNet,
    "kf_mlp": KFMLP,
    "kf_lenet": KFLeNet,
    "masked_mlp": MaskedMLP
}

ALL_MODELS_BASE_TYPE = {
    "mlp": "mlp",
    "lenet": "lenet",
    "kf_mlp": "mlp",
    "kf_lenet": "lenet",
    "masked_mlp": "mlp"
}


def get_model(name):
    # @name         : name of the model

    assert name in ALL_MODELS, "Model %s is not on defined." % name

    return ALL_MODELS[name]
