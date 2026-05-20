"""Experiment execution big-agent package."""

from agents.agent_registry import get_agent_boundary

BOUNDARY = get_agent_boundary("experiment_execution")

__all__ = ["BOUNDARY"]

