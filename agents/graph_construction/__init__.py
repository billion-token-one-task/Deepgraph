"""Graph construction big-agent package."""

from agents.agent_registry import get_agent_boundary

BOUNDARY = get_agent_boundary("graph_construction")

__all__ = ["BOUNDARY"]

