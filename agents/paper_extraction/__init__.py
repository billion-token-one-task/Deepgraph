"""Paper extraction big-agent package."""

from agents.agent_registry import get_agent_boundary

BOUNDARY = get_agent_boundary("paper_extraction")

__all__ = ["BOUNDARY"]

