import importlib
from typing import Type, List

from .base_agent import BaseAgent
from .base_agent_wrappers import add_base_wrappers

ALL_AGENTS = {
    # "<config_name>": ("<module_name>", "<class_name>")
    "baseline": ("base_agent", "BaseAgent"),
    "ewc": ("ewc", "EWC"),
    "kf": ("kf_agent", "KFAgent"),
    "fim_ewc": ("fim_ewc", "FIMEWC"),
    "full_approx": ("full_approx", "FullApprox"),
    "seq_laplace": ("sequential_laplace_approximation", "SequentialLaplaceApproximation"),
    "andrei_test": ("andrei_test", "AndreiTest"),
    "sparse_kf": ("sparse_laplace_kfc", "SparseKFLaplace"),
    "test_constraint_importance": ("test_constraint_importance", "TestConstraintImportance"),
    "task_dependent_unit": ("task_dependent_unit", "TaskDependentUnitEWC"),
}


def get_agent(name: str, base_wrappers: List[str] = list()) -> Type:
    # @name         : name of agent
    # @wrappers         : list of name of wrappers

    assert name in ALL_AGENTS, "Agent %s is not on defined." % name

    module = importlib.import_module("agents." + ALL_AGENTS[name][0])
    agent = getattr(module, ALL_AGENTS[name][1])

    if len(base_wrappers) > 0:
        agent.__bases__ = add_base_wrappers(base_wrappers, agent.__bases__)

    return agent
