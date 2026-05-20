# DeepGraph + EvoScientist: Automated Research Discovery System

## What This Is

An end-to-end system that reads thousands of ML papers from arXiv, builds a knowledge graph, automatically discovers genuine research opportunities (not "test X on Y" gaps), ranks them by paradigm-breaking potential, and on demand generates complete research plans with experiment design, success criteria, and paper outlines.

## Architecture

```
arXiv Papers
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                    DeepGraph                          │
│                                                      │
│  Ingestion ──► Extraction ──► Taxonomy ──► Knowledge │
│  (arXiv API)   (claims,       (auto-expand  Graph    │
│                 results,       312→470       (entities│
│                 methods)       nodes)        relations│
│                                              clusters)│
│       │                                              │
│       ▼                                              │
│  Insight Agent ──► Insight Ranker                    │
│  (cross-paper     (paradigm-breaking                 │
│   reasoning:       score 1-10,                       │
│   contradictions,  rationale)                        │
│   method transfers,                                  │
│   assumption                                         │
│   challenges)                                        │
│       │                                              │
│       ▼                                              │
│  Dashboard (Flask + vanilla JS)                      │
│  - Overview with live stats                          │
│  - Explore: navigate taxonomy, see insights per node │
│  - Insights: all 123 insights, filter/sort/rank      │
│  - Papers / Evidence / Feed / Providers              │
└──────────────┬───────────────────────────────────────┘
               │
               │  "Generate Full Plan" button
               ▼
┌──────────────────────────────────────────────────────┐
│                  EvoScientist                         │
│                                                      │
│  Research Bridge: gathers 30 papers + 40 claims +    │
│  contradictions + limitations → 18KB research        │
│  proposal                                            │
│       │                                              │
│       ▼                                              │
│  6 Sub-Agents:                                       │
│  Planner → Research → Code → Debug → Analysis →      │
│  Writing                                             │
│       │                                              │
│       ▼                                              │
│  Output:                                             │
│  - final_report.md (16KB full research plan)         │
│  - todos.md (staged execution checklist)             │
│  - Experiment design with success criteria           │
│  - Risk analysis + fallback plans                    │
│  - Paper outline with title + abstract               │
└──────────────────────────────────────────────────────┘
```

## End-to-End Tasks

### 1. Automated Paper Ingestion & Knowledge Extraction
```bash
# Pull 100 papers from arXiv and extract everything
curl -X POST http://localhost:8080/api/start \
  -H 'Content-Type: application/json' \
  -d '{"max_papers": 100}'
```
**What happens:** Fetches papers → downloads PDFs → extracts claims, results, methods → classifies into taxonomy → detects contradictions → builds knowledge graph. 3 LLM providers run in parallel (tabcode/kimi/minimax).

**Output:** Per paper: ~50 result tuples, claims with exact numbers, method descriptions, taxonomy assignments, entity-relation graph entries.

### 2. Research Insight Discovery
```bash
python3.12 -c "
from db.database import init_db; init_db()
from agents.insight_agent import discover_insights, store_insight
insights, tokens = discover_insights('ml.dl.nlp')
for ins in insights:
    store_insight(ins)
    print(f'[{ins[\"type\"]}] {ins[\"title\"]}')"
```
**What happens:** For each taxonomy node with 10+ papers, the insight agent does cross-paper reasoning to find:
- **Contradiction Analysis** — two papers disagree, propose hypothesis + experiment
- **Method Transfer** — technique from area A solves problem in area B
- **Assumption Challenge** — widely-held belief is wrong, with evidence
- **Ignored Limitation** — 3+ papers share same limitation nobody addresses
- **Paradigm Exhaustion** — diminishing returns signal need for new approach

**Output:** Insight with hypothesis, evidence, experiment design, expected impact, novelty/feasibility scores.

### 3. Paradigm-Breaking Ranking
```bash
python3.12 -c "
from db.database import init_db; init_db()
from agents.insight_ranker import rank_insights_batch
stats = rank_insights_batch()
print(stats)"
```
**What happens:** All insights are evaluated on a 1-10 paradigm-breaking scale:
- 1-3: Incremental
- 4-6: Meaningful but within current paradigms
- 7-8: Challenges fundamental assumptions
- 9-10: Reveals a paradigm is structurally wrong

**Output:** Each insight gets `paradigm_score` + `rank_rationale` stored in DB.

### 4. Generate Full Research Plan (on demand)
```bash
# Preview what would be sent
curl http://localhost:8080/api/research/proposal/6

# Launch EvoScientist to generate full plan
curl -X POST http://localhost:8080/api/research/launch \
  -H 'Content-Type: application/json' \
  -d '{"insight_id": 6}'
```
**What happens:**
1. Research Bridge gathers all supporting evidence (30 papers, 40 claims, contradictions, limitations)
2. Formats into 18KB structured research proposal
3. EvoScientist reads it with GPT-5.4
4. 6 sub-agents collaborate: Planner designs stages, Research validates novelty, Writing drafts outline

**Output in `~/research/insight_N_*/`:**
- `final_report.md` — Complete research plan with:
  - Novelty validation (is this actually new?)
  - 4-stage experiment design with exact models, datasets, metrics
  - Quantitative success criteria (e.g., "≥3 absolute points improvement")
  - Compute estimates (e.g., "600-900 A100 GPU hours")
  - 5 risks + fallback plans
  - Paper outline with title options + abstract sketch
- `todos.md` — Staged execution checklist
- `research_proposal.md` — Input proposal with all evidence

### 5. Paradigm Break Analysis
```bash
# Generate evidence file from all contradictions + assumption challenges
python3.12 agents/build_paradigm_evidence.py

# Launch EvoScientist for deep analysis
CUSTOM_OPENAI_USE_RESPONSES_API=true EvoSci \
  --workdir ~/research/paradigm_break \
  --auto-approve --ui cli \
  -p "Read evidence.md. Find where ML paradigms are BROKEN."
```
**Output:** Report identifying paradigm-level failures across ML, e.g.:
- Proxy optimization ≠ real capability
- Sequence models without persistent state can't solve long-horizon reality
- "More deliberation = better reasoning" is wrong
- Scaling laws break in scaffolded systems
- Robustness methods learn support coverage, not invariants

### 6. Dashboard Exploration
```
https://<tunnel-url>/
```
**Tabs:**
- **Overview** — Live stats, recently discovered insights, taxonomy map
- **Explore** — Navigate 470 taxonomy nodes, see per-node insights with evidence
- **Insights** — All 123 insights, filter by type, sort by paradigm score, launch research
- **Evidence** — Knowledge graph entities and relations
- **Papers** — Browse all ingested papers
- **Providers** — Monitor 3 LLM providers (tabcode/kimi/minimax)

## Setup

### Prerequisites
- Python 3.12+
- EvoScientist installed (`pip install EvoScientist` or from source)
- At least one LLM API key (OpenAI-compatible)

### Install
```bash
cd deepgraph
pip install -r requirements.txt
```

### Configure
Edit `config.py`:
- `LLM_BASE_URL` — your OpenAI-compatible API endpoint
- `LLM_API_KEY` — API key
- `LLM_MODEL` — model name (e.g., `gpt-5.4`)
- `ARXIV_CATEGORIES` — which arXiv categories to monitor

For EvoScientist:
```bash
EvoSci onboard  # interactive setup
# or edit ~/.config/evoscientist/config.yaml
```

### Run
```bash
python3.12 main.py  # starts dashboard on :8080

# Optional: expose via cloudflare tunnel
cloudflared tunnel --url http://localhost:8080
```

## Current Stats (as of 2026-03-21)
| Metric | Value |
|--------|-------|
| Papers ingested | 2,790 |
| Papers processed | 2,566 |
| Result tuples | 114,551 |
| Taxonomy nodes | 470 |
| Research insights | 123 |
| Contradictions | 34 |
| Knowledge graph entities | 47,837 |
| Knowledge graph relations | 276,999 |
| LLM tokens consumed | 38.6M |

## Key Files
| File | Purpose |
|------|---------|
| `main.py` | Entry point, starts Flask server |
| `config.py` | All configuration |
| `agents/insight_agent.py` | Cross-paper reasoning for research insights |
| `agents/insight_ranker.py` | Paradigm-breaking score ranking |
| `agents/research_bridge.py` | DeepGraph → EvoScientist bridge |
| `agents/llm_client.py` | Multi-provider LLM client with failover |
| `agents/extraction_agent.py` | Paper → structured claims/results |
| `orchestrator/pipeline.py` | Full pipeline orchestration |
| `db/database.py` | SQLite database layer |
| `db/taxonomy.py` | Hierarchical taxonomy management |
| `web/app.py` | Flask API + routes |
| `web/static/js/app.js` | Dashboard frontend |
