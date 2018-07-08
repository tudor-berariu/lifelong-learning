import importlib
from typing import Type, List

ALL_BASE_WRAPPERS = {
    # "<config_name>": ("<module_name>", "<class_name>")
    "sparsity": ("sparsity", "Sparsity"),
}


def add_base_wrappers(wrappers: List[str], start_base: Type) -> Type:
    # @wrappers         : list of name of wrappers

    for wrapper_name in reversed(wrappers):
        assert wrapper_name in ALL_BASE_WRAPPERS, f"Base agent wrapper {wrapper_name} not defined"

        module = importlib.import_module("agents.base_agent_wrappers." +
                                         ALL_BASE_WRAPPERS[wrapper_name][0])
        wrapper = getattr(module, ALL_BASE_WRAPPERS[wrapper_name][1])

        wrapper.__bases__ = start_base
        start_base = (wrapper, )

    return start_base
