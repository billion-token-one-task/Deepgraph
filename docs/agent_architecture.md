# DeepGraph Agent Architecture

DeepGraph now has a compatibility-first big-agent layout. The legacy module
paths remain valid, while the folders below define ownership boundaries for
future moves and new code.

| Big agent folder | Responsibility |
| --- | --- |
| `agents/paper_extraction/` | Paper discovery, PDF parsing, extraction, grounding, and source completeness. |
| `agents/graph_construction/` | Evidence graph, taxonomy, graph signals, and graph feedback loop. |
| `agents/idea_generation/` | Research idea generation, ranking, routing, and novelty checks. |
| `agents/experiment_planning/` | Benchmark contracts, experiment scaffolds, reviews, and artifact audits. |
| `agents/experiment_execution/` | Validation loops, GPU execution, remote shards, health checks, and merge watchers. |
| `agents/manuscript_generation/` | Manuscripts, figures, literature discovery, refinement, and submission bundles. |
| `agents/orchestration/` | End-to-end coordination, background jobs, workspace layout, web service, and deployment hooks. |

The executable map is `agents/agent_registry.py`. It records each big agent's
legacy modules, relevant scripts, and `deepgraph.toml` configuration sections.

Configuration defaults now live in root `deepgraph.toml`. Runtime environment
variables and `.env` still override TOML values, so existing deployments and
long-running jobs keep their behavior.

