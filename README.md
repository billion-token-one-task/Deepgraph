# DeepGraph

DeepGraph is an open scientific discovery engine. It ingests papers, extracts structured evidence, builds a knowledge graph, and runs a closed-loop pipeline that generates research hypotheses, tests them through autonomous experiments, and feeds results back into the graph.

## What It Does

DeepGraph answers three questions:

1. **What is this research area about?** — plain-language overviews, key methods, datasets, and entities for any taxonomy node.
2. **What are people not solving yet?** — opportunity themes grounded in paper limitations, contradictions, and sparse evidence regions.
3. **What should we try next?** — cross-field structural insights (Tier 1) and executable paper-ready ideas (Tier 2), validated through autonomous experiments.

## Architecture

```
Papers (arXiv)
  │
  ▼
Ingestion ─── PDF parse ─── LLM extraction
  │
  ▼
Knowledge Graph (entities, relations, claims, evidence)
  │
  ├──► Domain Summaries & Opportunity Briefs
  │
  ▼
Signal Harvester (SQL-based, zero LLM cost)
  │  cross-node overlap, convergent patterns,
  │  contradiction clusters, performance plateaus
  │
  ├──► Tier 1: Paradigm Agent
  │      structural isomorphisms across distant subfields
  │
  ├──► Tier 2: Paper Idea Agent
  │      executable top-venue paper ideas
  │
  ▼
Experiment Forge → Validation Loop
  │  scaffold experiments, run baselines,
  │  test hypotheses, interpret results
  │
  ▼
Knowledge Loop ◄── Meta-Learner
  feed results back into graph,     re-weight signal
  cascade hypothesis updates        harvesting strategy
```

### Core Components

| Directory | Purpose |
|-----------|---------|
| `ingestion/` | arXiv paper discovery and PDF parsing |
| `agents/` | LLM extraction, insight generation, experiment orchestration |
| `db/` | Schema, taxonomy, evidence graph, entity resolution |
| `orchestrator/` | End-to-end pipeline and background discovery scheduler |
| `web/` | Flask API and interactive dashboard |

### Big Agent Boundaries

The project is organized around compatibility-first big-agent folders. Existing
module imports stay valid; new code should use these folders as ownership
boundaries.

| Big agent folder | Purpose |
|------------------|---------|
| `agents/paper_extraction/` | Paper discovery, PDF parsing, extraction, grounding, and source completeness |
| `agents/graph_construction/` | Evidence graph, taxonomy growth, graph signals, and feedback loop |
| `agents/idea_generation/` | Insight generation, ranking, reasoning, novelty checks, and idea routing |
| `agents/experiment_planning/` | Benchmark contracts, experiment scaffolding, reviews, and artifact audits |
| `agents/experiment_execution/` | Validation loops, GPU jobs, remote shards, health checks, and merge watchers |
| `agents/manuscript_generation/` | Manuscripts, figures, literature discovery, refinement, and bundles |
| `agents/orchestration/` | End-to-end scheduling, workspace layout, web service, and deployment hooks |

See `agents/agent_registry.py` and `docs/agent_architecture.md` for the exact
legacy module map.

### Agent Modules

**Extraction & Analysis**
- `extraction_agent` — classify papers and extract structured results
- `insight_agent` — deep cross-paper reasoning (contradictions, method transfers, paradigm shifts)
- `insight_ranker` — rank and prioritize insights
- `reasoning_agent` — multi-step reasoning chains
- `abstraction_agent` — abstract structural patterns
- `domain_summary_agent` — plain-language node summaries
- `research_bridge` — connect findings across domains
- `taxonomy_expander` — grow taxonomy from discovered structure

**Discovery Pipeline (SciForge)**
- `signal_harvester` — SQL-based cross-field signal detection
- `paradigm_agent` — Tier 1: discover hidden unifying structures across subfields
- `paper_idea_agent` — Tier 2: generate executable top-venue paper ideas
- `novelty_verifier` — check if insights already exist in literature
- `experiment_forge` — translate insights into runnable experiments
- `validation_loop` — hypothesis-directed experiment engine
- `result_interpreter` — parse outcomes into structured verdicts
- `knowledge_loop` — feed results back into knowledge graph
- `meta_learner` — self-improve discovery strategy from experimental history

## Quick Start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API key
export $(grep -v '^#' .env | xargs)
python3.12 main.py
```

Then open `http://localhost:8080`.

## Configuration

Default configuration lives in `deepgraph.toml`. Runtime environment variables
and `.env` still override TOML values, which keeps existing deployments and
long-running jobs compatible.

Key override variables:

| Variable | Description |
|----------|-------------|
| `DEEPGRAPH_LLM_API_KEY` | Required. LLM API key for extraction and generation |
| `DEEPGRAPH_LLM_SECONDARY_*` | Optional second OpenAI-compatible route for parallel LLM calls |
| `DEEPGRAPH_LLM_EXTRA_PROVIDERS_JSON` | Optional JSON list of additional OpenAI-compatible routes |
| `DEEPGRAPH_PROFILE` | `machine_learning` or `open_science` |
| `DEEPGRAPH_ROOT_NODE_ID` | Defaults to `ml` or `science` based on profile |
| `DEEPGRAPH_ARXIV_CATEGORIES` | Optional comma-separated arXiv category override |
| `DEEPGRAPH_BACKFILL_GRAPH_ON_START` | Backfill graph from existing structured records at startup |
| `DEEPGRAPH_WEB_PORT` | Dashboard port (default 8080) |

Switch to the broader science profile:

```bash
export DEEPGRAPH_PROFILE=open_science
export DEEPGRAPH_ROOT_NODE_ID=science
python3.12 main.py
```

### Discovery Pipeline Configuration

The SciForge discovery pipeline has additional tuning knobs via `DISCOVERY_BULK_*` environment variables — see [config.py](config.py) for the full list.

## Manuscript Venue Routing

When `paper_idea_agent` decides an insight is worth a paper, the manuscript pipeline routes it through a per-venue adapter chain rather than a single hard-coded ICLR template. The routing stack:

1. `agents/venue_router.py` reads `manuscript_venues/venues_v1.yaml` (6 venues: `iclr2026`, `neurips2024`, `icml2024`, `acl_arr`, `cvpr2024`, `arxiv_plain`) and picks a primary + secondary based on subject area, deadline window, and submission_mode.
2. `agents/manuscript_templates/` resolves the choice into a `TemplateAdapter` (column layout, bibstyle, page budget, required packages) via `get_adapter(template_id)`.
3. `agents/format_linter.py` runs 12 checks against the rendered LaTeX — 7 structural plus the 5 mandated by issue #14 (`font_size_consistency`, `section_spacing`, `float_density`, `citation_density`, `bib_style_match`); a failure blocks the submission gate.
4. `agents/paper_orchestra_pipeline.py` calls `require_submission_ready()` so synthetic data never reaches a manuscript bundle.
5. The Flask routes under `web/manuscript_routes.py` expose `/api/manuscript/route` and the dashboard panel for human review.

Entry point: pass `venue_hint=` to `paper_orchestra_pipeline.run(...)` or set `DEEPGRAPH_DEFAULT_VENUE` in `.env`. See `docs/top_venue_manuscript_chain.md` for the full router → adapter → linter → gate diagram.

## Science Taxonomy

The `open_science` profile spans:

- Mathematics & Statistics
- Physics
- Chemistry & Materials
- Life Sciences
- Medicine & Health
- Earth & Climate
- Engineering
- Computing & AI

## Packaging

```bash
python3.12 -m pip install build
python3.12 -m build
```

## Running Tests

```bash
python3.12 -m unittest discover -s tests
```

## Data & Security

Large local artifacts (SQLite databases, WAL files, cached PDFs, logs) are excluded by `.gitignore`. The open-source version does not hardcode API keys — credentials are provided through environment variables only.

## Status

DeepGraph has evolved from a passive literature analysis tool into an active discovery system. Current strengths:

- Literature ingestion and evidence extraction
- Entity/relation/evidence graph with auditable entity resolution
- Plain-language node summaries and opportunity surfacing
- **Closed-loop discovery**: signal harvesting → insight generation → autonomous experiment → knowledge feedback
- Meta-learning from experimental track record

Still improving:

- Entity canonicalization across papers
- Cross-source deduplication
- Richer scientific ontologies beyond built-in taxonomy packs
- Large-scale historical backfills

## License

MIT
