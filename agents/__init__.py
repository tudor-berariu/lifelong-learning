import importlib
from typing import Type

from .base_agent import BaseAgent

ALL_AGENTS = {
    "baseline": ("base_agent", "BaseAgent"),
}


def get_agent(name) -> Type:
    # @name         : name of agent

    assert name in ALL_AGENTS, "Agent %s is not on defined." % name

    module = importlib.import_module("agents." + ALL_AGENTS[name][0])
    agent = getattr(module, ALL_AGENTS[name][1])

    return agent
