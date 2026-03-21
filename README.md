# DeepGraph

DeepGraph is an open scientific opportunity mapping engine. It ingests papers, extracts structured evidence, generates plain-language domain briefs for non-specialists, and highlights open gaps worth exploring.

The current codebase supports two operating modes:

- `machine_learning`: optimized for the existing ML-focused taxonomy and current demo data.
- `open_science`: a broader science-oriented taxonomy spanning computing, biology, medicine, physics, chemistry, Earth science, mathematics, and engineering.

## What It Does

DeepGraph is not just a paper browser. It tries to answer two practical questions:

1. What is this research area roughly about?
2. What are people not solving yet?

For each taxonomy node it can produce:

- A plain-language overview for non-specialists
- A short explanation of why the area matters
- A summary of the main workstreams people are building
- Common methods, datasets, and recurring patterns
- Core entities and the relations connecting them
- Opportunity themes and open questions grounded in paper limitations
- Evidence tables for methods, datasets, and metrics

## Architecture

The pipeline is:

1. Discover papers from arXiv
2. Download PDFs and extract text
3. Ask an LLM to classify papers and extract results
4. Store paper-level briefs, claims, methods, entities, relations, and evidence rows
5. Build node-level graph summaries, summaries, and opportunity briefs
6. Serve an interactive dashboard

Core components:

- `ingestion/`: paper discovery and PDF parsing
- `agents/`: LLM extraction, contradiction detection, and domain-summary generation
- `db/`: schema, taxonomy, summary caching, and matrix building
- `orchestrator/`: end-to-end pipeline
- `web/`: Flask API and dashboard

## Quick Start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export $(grep -v '^#' .env | xargs)
python3.12 main.py
```

Then open `http://localhost:8080`.

## Configuration

DeepGraph is configured through environment variables.

Important ones:

- `DEEPGRAPH_LLM_API_KEY`: required for extraction and summary generation
- `DEEPGRAPH_PROFILE`: `machine_learning` or `open_science`
- `DEEPGRAPH_ROOT_NODE_ID`: defaults to `ml` for ML mode and `science` for open science mode
- `DEEPGRAPH_ARXIV_CATEGORIES`: optional comma-separated override
- `DEEPGRAPH_BACKFILL_GRAPH_ON_START`: backfill graph entities and relations from existing structured records at startup
- `DEEPGRAPH_WEB_PORT`: dashboard port

Example: switch to the broader science profile

```bash
export DEEPGRAPH_PROFILE=open_science
export DEEPGRAPH_ROOT_NODE_ID=science
python3.12 main.py
```

## Open Science Taxonomy

The `open_science` profile includes a broader root taxonomy with these major branches:

- Mathematics & Statistics
- Physics
- Chemistry & Materials
- Life Sciences
- Medicine & Health
- Earth & Climate
- Engineering
- Computing & AI

This gives you a cleaner path toward a general scientific knowledge graph instead of a hardcoded ML-only tree.

## Packaging

The project includes a `pyproject.toml`, so you can build source and wheel artifacts with:

```bash
python3.12 -m pip install build
python3.12 -m build
```

Artifacts will be written to `dist/`.

## Running Tests

```bash
python3.12 -m unittest discover -s tests
```

## Notes On Data

Large local artifacts are intentionally not meant for source control:

- SQLite databases
- WAL files
- cached PDFs
- logs

They are excluded by `.gitignore`.

## Security

The open-source version does not hardcode API keys. You must provide credentials through environment variables.

## Status

This repository is still an early-stage research system. The current implementation is strongest at:

- literature ingestion
- evidence extraction
- entity/relation/evidence graph storage
- auditable entity resolution and merge candidates
- plain-language node summaries
- opportunity surfacing from limitations and sparse evidence regions

It also now includes a deterministic graph backfill path for older databases:

- existing `methods`, `results`, `claims`, and `paper_insights` can be converted into graph entities and relations
- node-level graph summaries can be regenerated without rerunning the full LLM extraction pipeline
- similar entities can be surfaced as merge candidates for manual review instead of being destructively merged

It is still weak at:

- entity canonicalization across papers
- cross-source deduplication
- strong scientific ontologies beyond the built-in taxonomy packs
- large-scale historical backfills across all of science

## License

MIT
