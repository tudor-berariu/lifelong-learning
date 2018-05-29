import importlib

ALL_ALGORITHMS = {
    "baseline": "baseline_seq",
    "ewc": ".ewc"
}


def get_algorithm(name):
    # @name         : name of algorithm

    assert name in ALL_ALGORITHMS, "Algorithm %s is not on defined." % name

    module = importlib.import_module("algorithms." + ALL_ALGORITHMS[name])
    train_sequentially = getattr(module, "train_sequentially")

    return train_sequentially
