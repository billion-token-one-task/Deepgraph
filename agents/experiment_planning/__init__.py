"""Experiment planning big-agent package."""

from agents.agent_registry import get_agent_boundary

BOUNDARY = get_agent_boundary("experiment_planning")

__all__ = ["BOUNDARY"]

