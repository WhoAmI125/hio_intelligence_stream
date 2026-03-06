"""
Scenario Module for Multi-Scenario Decomposition

Based on EVA Q2E approach: decomposing complex multi-task detection
into focused single-scenario agents for improved accuracy.

Reference: https://mellerikat.com/blog_tech/Research/multi_scenario
"""

from .prompts import (
    CASH_SCENARIO,
    VIOLENCE_SCENARIO,
    FIRE_SCENARIO,
    get_scenario_prompt,
    ScenarioType
)
from .base_scenario import BaseScenario, ScenarioResult

__all__ = [
    'CASH_SCENARIO',
    'VIOLENCE_SCENARIO',
    'FIRE_SCENARIO',
    'get_scenario_prompt',
    'ScenarioType',
    'BaseScenario',
    'ScenarioResult'
]
