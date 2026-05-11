# Idea Generation Agent

Owns insight generation, ranking, reasoning, novelty checks, and routing ideas toward experiments.

Primary legacy modules:

- `agents.insight_agent`
- `agents.insight_ranker`
- `agents.reasoning_agent`
- `agents.abstraction_agent`
- `agents.research_bridge`
- `agents.paradigm_agent`
- `agents.paper_idea_agent`
- `agents.novelty_verifier`
- `agents.evidence_planner`
- `agents.idea_route`
- `agents.discovery_metadata`
- `agents.discovery_supervisor`

Configuration lives in `deepgraph.toml` under `discovery`, `idea`, `llm`, and `paper_orchestra`.

