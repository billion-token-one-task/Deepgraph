# Graph Construction Agent

Owns evidence graph construction, taxonomy growth, graph-derived signals, and the graph feedback loop.

Primary legacy modules:

- `db.evidence_graph`
- `db.opportunity_engine`
- `db.taxonomy`
- `agents.taxonomy_expander`
- `agents.domain_summary_agent`
- `agents.signal_harvester`
- `agents.knowledge_loop`
- `agents.meta_learner`

Configuration lives in `deepgraph.toml` under `database`, `graph`, `discovery`, `pipeline`, and `paths`.

