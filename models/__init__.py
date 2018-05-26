from .mlp import MLP

ALL_MODELS = {
    "mlp": MLP,
}


def get_model(name):
    # @name         : name of the model

    assert name in ALL_MODELS, "Model %s is not on defined." % name

    return ALL_MODELS[name]
