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

## Meta-harness v1 (0.2.0)

This release adds a scoped research-control plane for reproducible autonomous
experiments. The important change is that every costly action now requires a
short-lived `ResourceGrant` tied to an Agenda and idea; measured usage and
artifacts settle the grant before an outcome is recorded.

Key innovations:

- Agenda-scoped Frontier/Decision/Grant/Outcome contracts with hard token/GPU
  caps and fail-closed legacy paths.
- Durable PostgreSQL claims and idempotency for CPU and SSH GPU jobs, including
  restart recovery and truthful unknown-submission quarantine.
- Role-separated, metered LLM routing with durable provider cooldowns.
- Hash-pinned bubblewrap held-in/held-out/canary evaluation plus signed review
  approval; Colab is implemented but excluded from the 0.2.0 release scope.

### CPU + SSH A100 quickstart

1. Run the normal setup above and apply the add-only PostgreSQL migration to a
   disposable database.
2. Configure only secret references; never put keys or passwords in TOML:

   ```bash
   export DEEPGRAPH_COMPUTE_SSH_TARGET_REF=env:DEEPGRAPH_SSH_TARGET
   export DEEPGRAPH_COMPUTE_SSH_CREDENTIAL_REF=env:DEEPGRAPH_SSH_CREDENTIAL
   export DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN='replace-with-a-short-lived-token'
   ```

3. Create an Agenda and short-lived `ResourceGrant`, then submit work through
   the scoped API or scheduler. The scheduler refuses missing grants,
   unlimited budgets, unscoped backlog and unknown backend outcomes.

See [docs/integration/CONFIGURATION.md](docs/integration/CONFIGURATION.md) and
[docs/integration/ACCEPTANCE_EVIDENCE.md](docs/integration/ACCEPTANCE_EVIDENCE.md)
for the complete contract and the accepted CPU + SSH A100 scope.

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

## Changelog

### 2026-06-26

- 当前版本已经打通完整的端到端论文生产链路：从文献摄取、证据抽取、知识图谱构建、研究想法生成、实验规划与执行，到结果解释、图表生成、LaTeX 组稿和论文初稿输出，系统已经能够形成可审阅的完整论文包。
- 当前论文质量已经达到 workshop 水平：系统能够给出清晰的问题动机、实验结果、方法描述和初步论证，也能生成基本完整的图表、引用和实验报告。但距离 C 会论文仍有差距，主要体现在选题锋利度、贡献压缩、实验说服力、审稿防御、写作稳定性和最终 PDF 观感上。
- 下一阶段的目标将从“能端到端产出论文”切换到“稳定产出更高质量论文”。核心方法是建立论文质量闭环：每次生成论文后，都要经过自动审稿、人工抽查、问题归因、prompt/agent/实验协议修订，再把这些修订沉淀成可复用的质量门控。

#### Immediate three-week plan

**Week 1: Problem awareness and contribution framing**

Goal: make every generated paper answer three questions within the abstract and first two introduction paragraphs: what problem is being solved, why the problem matters, and what the method contributes beyond obvious baselines.

Concrete work:

1. Add a problem-first gate before manuscript generation. The gate must reject papers whose motivation is only a broad field description, whose claimed gap is not backed by evidence, or whose contribution is a raw experiment result rather than a clear research claim.
2. Require each paper to produce a `paper_contract.json` before writing. It should lock the target venue, problem statement, main claim, forbidden overclaims, required experiments, expected figures, and terminology.
3. Add a title and abstract critic. The critic should score whether the title is memorable, whether the abstract states the result quantitatively, and whether the contribution is specific enough to survive reviewer scrutiny.
4. Build a small benchmark set of strong workshop/C-conference papers in adjacent areas. Use them as style and structure references, not as text to copy.
5. Add a regression test that fails if generated papers use empty phrases such as "significantly improves", "novel framework", or "comprehensive evaluation" without concrete evidence nearby.

Deliverables:

- `paper_contract.json` becomes mandatory for every manuscript run.
- Each generated paper includes a one-page problem/contribution brief before full drafting.
- The manuscript reviewer reports separate scores for problem clarity, contribution sharpness, and claim-evidence alignment.

Acceptance criteria:

- At least 3 generated papers can explain their problem, method, and result in one paragraph without relying on vague claims.
- The introduction contains explicit motivation, gap, method, and contribution statements.
- Every major claim in the abstract maps to a completed experiment or literature-backed evidence item.

**Week 2: Evidence, experiments, and reviewer-grade validation**

Goal: make the experimental section strong enough that a reviewer can understand what was tested, why the baselines are fair, and whether the result is robust.

Concrete work:

1. Define a standard experiment matrix for each generated idea: main comparison, ablation, sensitivity analysis, failure case, cost/latency analysis, and at least one negative or boundary condition.
2. Require every experiment to emit machine-readable artifacts: metrics table, run configuration, seed list, dataset split, baseline description, and failure logs.
3. Add an evidence completeness auditor. It should block manuscript claims when the required experiment is missing, incomplete, too small, or only a smoke test.
4. Add baseline fairness checks: same dataset, same budget, comparable model size or compute, same evaluation protocol, and clearly stated differences when exact parity is impossible.
5. Add statistical reporting where applicable: multiple seeds, confidence intervals, standard deviation, paired tests, or bootstrap intervals.
6. Improve figure generation so that every paper has at least one main result figure, one ablation or sensitivity figure, and one compact method or pipeline figure.

Deliverables:

- Each paper has an `evidence_manifest.json`, `claim_evidence_matrix.json`, and `experiment_judgement.json`.
- Experiment tables include baselines, ablations, seeds, and cost/latency where relevant.
- The paper cannot be marked submission-ready unless evidence gates pass.

Acceptance criteria:

- No abstract or introduction claim is unsupported by a completed result.
- Main experimental results can be reproduced from saved configs and artifacts.
- Reviewer-style questions such as "what happens without this component?" and "is the baseline fair?" are answered in the paper or appendix.

**Week 3: Writing, figures, revision, and reviewer defense**

Goal: make the final paper read less like a generated draft and more like a polished conference submission.

Concrete work:

1. Add a section-level writing critic for abstract, introduction, related work, method, experiments, limitations, and conclusion.
2. Add a reviewer simulation loop with at least three personas: skeptical method reviewer, experimental rigor reviewer, and clarity/positioning reviewer.
3. Require revision passes to produce a structured diff: what claim changed, what evidence was added, what wording was tightened, and which reviewer objection was addressed.
4. Add figure and table layout checks: captions must state the takeaway, axes must be interpretable, tables must fit the page, and every visual must be referenced in text.
5. Add PDF sanity checks before declaring success: compilation, page budget, missing references, broken citations, overfull boxes, unreadable figures, and appendix consistency.
6. Build a final submission checklist covering anonymity, citation validity, artifact availability, reproducibility notes, limitations, ethics where relevant, and venue formatting.

Deliverables:

- `reviewer_report.json` with scored objections and required fixes.
- `manuscript_revision_history.json` recording each revision pass.
- `submission_checklist.md` generated with pass/fail status.
- A final PDF that compiles cleanly and is visually inspectable.

Acceptance criteria:

- The system can complete at least two revision rounds without drifting away from the original evidence.
- The final manuscript has no unsupported headline claims, no missing citations, and no obvious formatting failures.
- Figures and tables communicate their takeaways without requiring the reader to inspect raw logs.

#### Eighteen-week quality roadmap

The 18-week roadmap is organized as six three-week cycles. Each cycle should end with generated papers, reviewer reports, failure analysis, and code/prompt changes that permanently improve the next cycle.

1. Weeks 1-3: Establish the quality gate foundation. Make problem contracts, evidence manifests, reviewer simulation, and PDF checks mandatory.
2. Weeks 4-6: Improve idea selection. Rank ideas by novelty, tractability, evidence availability, benchmark clarity, and expected reviewer interest before spending experiment budget.
3. Weeks 7-9: Improve experimental depth. Expand benchmark harnesses, strengthen baselines, add ablations by default, and standardize statistical reporting.
4. Weeks 10-12: Improve manuscript polish. Tune section-specific writing agents, figure captions, related work positioning, limitation writing, and contribution compression.
5. Weeks 13-15: Improve automated review and revision. Train the system to convert reviewer objections into concrete experiment additions, claim edits, and figure/table changes.
6. Weeks 16-18: Submission rehearsal. Generate several candidate papers end to end, run full internal review, select the strongest work, and iterate until it reaches credible C-conference submission quality.

#### Operating loop

Every manuscript run should follow the same loop:

1. Generate or select a research idea.
2. Produce `paper_contract.json` and reject weak ideas early.
3. Plan experiments and required evidence before writing.
4. Run experiments and save reproducible artifacts.
5. Build figures and tables from artifacts, not from prose alone.
6. Draft the paper with claim-evidence constraints.
7. Run automated reviewers and evidence auditors.
8. Revise the paper, experiments, figures, and claims.
9. Compile and visually inspect the PDF.
10. Record failures and feed them back into prompts, gates, tests, and agent policies.

The intended outcome is not one lucky good paper, but a repeatable quality engine: DeepGraph should learn which ideas are worth writing, which claims are defensible, which experiments persuade reviewers, and which writing patterns move a draft from workshop quality toward C-conference quality.

## License

MIT
