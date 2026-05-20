"""Idea generation big-agent package."""

from agents.agent_registry import get_agent_boundary

BOUNDARY = get_agent_boundary("idea_generation")

__all__ = ["BOUNDARY"]

