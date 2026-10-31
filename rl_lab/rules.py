"""Inject student functions into the provided training helpers; no answer fallback."""
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class LearningRules:
    epsilon_greedy: Callable
    td_target: Callable
    q_update: Callable
