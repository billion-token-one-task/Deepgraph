# DeepGraph

DeepGraph is an open research engine: it reads the literature at scale, builds
a structured evidence graph, proposes research directions from that graph, and
carries the promising ones through contracted, budgeted, auditable experiments
toward manuscript output. Around the engine sits a control plane that prices
every unit of spend and gates every promotion of a claim -- so what comes out
the other end can be trusted, reproduced, and defended.

> 中文导读: DeepGraph 是一个带控制平面的开放研究引擎。它规模化地读论文、抽取
> 结构化证据、构建知识图谱、从图谱信号中生成研究方向, 并把有希望的方向推进到
> 有合同、有预算、可审计的实验与论文产出。它的差异化在于治理: 每一笔开销都有
> 授权和结算, 每一次结论升级都要过闸门。

## What it does

1. **Ingest**: arXiv discovery, PDF parsing (GROBID), LLM extraction of claims,
   methods, results, and taxonomy into PostgreSQL.
2. **Graph**: entities, relations, contradictions, and evidence links merged
   across papers, with entity resolution and domain summaries.
3. **Discover**: a zero-LLM-cost SQL signal harvester finds overlaps,
   convergent patterns, contradiction clusters, and plateaus; paradigm and
   paper-idea agents turn them into executable research candidates.
4. **Execute**: candidates that pass the gates get a scoped resource grant, a
   frozen benchmark contract, and a run on CPU or remote GPU.
5. **Write**: manuscript generation under contract -- claim-evidence matrices,
   reviewer simulation, figure and layout audits, and PDF sanity checks.

## Why it is different

Most automated-research systems treat "the job finished" as "the result is
real". DeepGraph is built on the opposite premise, and that discipline is its
core innovation:

- **An evidence ladder with content-hash gates.** A result climbs
  `sanity_passed -> full_benchmark_complete -> evidence_audited -> decided`
  one step at a time; every step demands hashed raw artifacts, a pinned
  benchmark contract, and an independent evaluation, or it refuses to move
  (`meta_harness/evidence_state.py`).
- **Two registers, never merged.** "Did it run" and "what does the audited
  evidence say" are stored, served, and displayed separately, down to the
  dashboard badges. The system cannot dress process up as proof.
- **A topic gate that prices admission in bits.** Before a candidate may spend
  anything, it must carry a pre-registered prediction; expected information
  gain is computed as a pure function -- no LLM, no database, no override
  switch (`agents/topic_gate.py`).
- **Grant economics.** Every spend is reserve -> measure -> settle. Failures
  settle honestly and unused reservations are refunded to the agenda budget
  (`meta_harness/`). Nothing runs grantless.
- **Manuscript gates that can say no.** A paper claim must trace to a frozen
  contract, a complete evidence manifest, multi-seed statistics, and a signed
  reviewer approval before `manuscript_allowed`
  (`contracts/scientific_evidence.py`, `agents/paper_completeness.py`).
- **Operations as part of the contract.** SHA-pinned deployment manifests,
  additive-only migrations, a self-heal watchdog with single-action ticks, and
  read-only audit scripts (`orchestrator/selfheal_policy.py`, `deploy/`).

## Scale

Operational snapshot, 2026-08-06, from the live `/api/stats`:

- 21,151 papers collected, 6,729 fully processed
- 28,700 claims and 156,194 results extracted
- 248,441 graph entities, 735,913 relations, 5,051 taxonomy nodes
- 11,936 insights, 110 deep insights, 25,652 mapped opportunities
- 113 experiment runs, 17 GPU workers registered
- 921M LLM tokens invested

Demo walkthrough, screenshots, and case studies: [docs/SHOWCASE.md](docs/SHOWCASE.md).

## Quick start

Python 3.12+ and an LLM API key are the minimum. PostgreSQL enables the full
control plane.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # set at least DEEPGRAPH_LLM_API_KEY
export $(grep -v '^#' .env | xargs)
python3.12 main.py
```

Open `http://localhost:8080`.

Full deployment -- PostgreSQL migration, systemd units, remote GPU backends,
configuration reference: [docs/DEPLOY.md](docs/DEPLOY.md).

## Architecture

```
Papers (arXiv)
  |
  v
Grant-scoped ingestion -- PDF parse -- LLM extraction
  |
  v
Knowledge graph (entities, relations, claims, evidence)
  |
  +--> Domain summaries and opportunity briefs
  |
  v
Signal harvester (SQL-based, zero LLM cost)
  |
  +--> Tier 1: paradigm agent -- structural isomorphisms across subfields
  +--> Tier 2: paper idea agent -- executable paper ideas
  |
  v
Meta-harness control plane
  |   agenda scope, topic gate, portfolio decision,
  |   ResourceGrant, execution, settlement, evidence ladder
  |
  v
Experiments (CPU / remote GPU) --> audited results --> manuscripts
```

| Directory | Purpose |
|---|---|
| `contracts/` | Versioned record types and their validation rules |
| `meta_harness/` | Control plane: ladder, grants, authorities, repository |
| `ingestion/` | arXiv discovery and PDF parsing |
| `agents/` | Extraction, insight generation, gating, experiment orchestration |
| `db/` | Schema, migrations, taxonomy, evidence graph, entity resolution |
| `orchestrator/` | Scheduling, workers, self-heal policy, compute runtime |
| `web/` | Flask API, provenance API, dashboard |
| `deploy/` | Systemd units, Caddyfile, deployment manifests |
| `scripts/` | Operator CLIs, audits, migrations, manifest tooling |
| `plugins/` | Self-contained experiment harnesses (e.g. `examples/cggr`) |
| `tests/` | 90 test modules |

## Documentation

| Document | What it covers |
|---|---|
| [docs/DEPLOY.md](docs/DEPLOY.md) | Full deployment: database, systemd, GPU backends, configuration |
| [docs/SHOWCASE.md](docs/SHOWCASE.md) | Live numbers, demo route, case studies |
| [docs/ROADMAP.md](docs/ROADMAP.md) | What is next |
| [docs/upgrade-plan-v1-v2.md](docs/upgrade-plan-v1-v2.md) | The V1 -> V2 upgrade plan |
| [LATENT_COMMUNICATION_RESEARCH.md](LATENT_COMMUNICATION_RESEARCH.md) | The team's latent-communication research line |
| `docs/internal/` | Operator runbooks, state dictionary, configuration and architecture reference |

Release history: [CHANGELOG.md](CHANGELOG.md) (maintained separately).

## Tests

```bash
python3.12 -m unittest discover -s tests
```

Read-only audit scripts (scope, SQL, static, state-authority, LLM callers):

```bash
python3.12 scripts/meta_harness_scope_audit.py
python3.12 scripts/meta_harness_sql_audit.py
python3.12 scripts/meta_harness_static_audit.py
```

## Data and security

- No hardcoded credentials; secrets come from the environment as *references*,
  not stored values. SSH execution requires strict known-host pinning.
- Public API responses pass through a scrubber that strips path and log keys
  and redacts absolute paths; the SSE stream is scrubbed the same way
  (`web/app.py:84-106`).
- Every parameterless GET route is walked by a leak-guard test asserting no
  filesystem paths or log tails appear in any response
  (`tests/test_provenance_web.py:282`).

## License

See [LICENSE](LICENSE).
