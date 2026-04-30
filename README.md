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
  │  compile research specs, select capabilities,
  │  run single-script or benchmark-suite validation
  │
  ▼
Evidence Gate → Manuscript → AI Review → Follow-up Plan
  │  statistical report, publishability status,
  │  grounded report/candidate, required experiments
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
- `research_spec_compiler` — normalize insight titles into reusable research specs
- `capability_registry` — choose implemented offline experiment capabilities
- `benchmark_suite` — run configured benchmark harnesses and record results
- `statistical_reporter` — summarize multi-seed benchmark evidence
- `evidence_gate` — decide whether outputs are preliminary, need more experiments, or are paper candidates
- `result_interpreter` — parse outcomes into structured verdicts
- `knowledge_loop` — feed results back into knowledge graph
- `manuscript_writer` — generate grounded reports or evidence-gated paper candidates
- `ai_reviewer` — review manuscript artifacts without changing experiment results
- `review_planner` — turn reviewer-required experiments into follow-up plan artifacts
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

Key environment variables:

| Variable | Description |
|----------|-------------|
| `DEEPGRAPH_LLM_API_KEY` | Required. LLM API key for extraction and generation |
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

### SciForge Experiment Workflow

Current experiment execution reuses `deep_insights` and `experiment_runs`; it does not create a separate research-project API. The workflow is:

```text
deep_insight
  -> research_spec.json
  -> capability selection
  -> forge
  -> single-script validation or benchmark-suite validation
  -> statistical_report.json
  -> evidence_gate.json
  -> manuscript report or paper_candidate.md
  -> AI review
  -> followup_experiment_plan.json
```

The first implemented benchmark capability is offline grouped fairness classification. Safe RL/CMDP is registered as a planned capability but is not implemented yet.

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
- Auditable experiment artifacts: research specs, execution plans, metrics, benchmark results, statistics, logs, iteration summaries, and manifests
- Evidence-gated manuscript generation: preliminary reports, additional-experiment reports, negative results, and `paper_candidate.md` only when evidence allows it
- Structured AI review and follow-up experiment planning for generated manuscript packages
- Meta-learning from experimental track record

Still improving:

- Entity canonicalization across papers
- Cross-source deduplication
- Richer scientific ontologies beyond built-in taxonomy packs
- Large-scale historical backfills
- Richer LaTeX templates, SLURM/GPU execution backends, and multi-round rebuttal automation
- Additional real benchmark capabilities beyond the current offline fairness suite

Generated manuscript packages are grounded in stored paper evidence and experiment artifacts, but they are not a guarantee of publishability. Evidence gates, AI review, and human review are required before submission. Failed runs should be classified by root cause before changing thresholds, seeds, prompts, or configuration.

## License

MIT
