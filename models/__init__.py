from .mlp import MLP
from .lenet import LeNet

ALL_MODELS = {
    "mlp": MLP,
    "lenet": LeNet
}

ALL_MODELS_BASE_TYPE = {
    "mlp": "mlp",
    "lenet": "lenet"
}

def get_model(name):
    # @name         : name of the model

    assert name in ALL_MODELS, "Model %s is not on defined." % name

    return ALL_MODELS[name]
