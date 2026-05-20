# Manuscript Generation Agent

Owns manuscript generation, figure orchestration, literature discovery, refinement, and submission bundles.

Primary legacy modules:

- `agents.manuscript_pipeline`
- `agents.paper_orchestra_pipeline`
- `agents.paper_orchestra_prompts`
- `agents.figure_agent`
- `agents.paperorchestra.full_pipeline`
- `agents.paperorchestra.figure_orchestra`
- `agents.paperorchestra.literature_discovery`
- `agents.paperorchestra.plotting_orchestra`
- `agents.paperorchestra.refinement_loop`
- `agents.paperorchestra.semantic_scholar`

Primary scripts:

- `scripts.paperbanana_wrapper`
- `scripts.repair_manuscript_artifacts`

Configuration lives in `deepgraph.toml` under `manuscript`, `paper_orchestra`, `llm`, and `paths`.

