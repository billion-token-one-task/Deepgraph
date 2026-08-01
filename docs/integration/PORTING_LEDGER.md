# meta-harness-v1 porting ledger

Status values: `planned`, `in_progress`, `ported`, `adapted`, `rejected`,
`deferred`, `needs_ci`.

Every row is a semantic-port decision. Source commits are immutable objects
listed in [BASELINE.md](BASELINE.md); no row authorizes a history merge.

| Phase | Source commit | Source path/symbol | Target | Behavioral contract | Verification | Status / non-port reason |
|---|---|---|---|---|---|---|
| 0 | all three | refs, schema and state strings | `docs/integration/*` | reproducible audit basis and cross-line vocabulary | refs, diff-stat, Git/AST/text | adapted |
| 0 | `bc74806` | candidate ancestry audit | no target merge | commit is already an ancestor of `integration/meta-harness-v1`; do not duplicate it | `merge-base --is-ancestor` passed | adapted |
| 0 | `9d24d29` | topic-gate semantic ledger | existing agenda/evidence/frontier contracts | no common history; semantic behavior may be ported only, never a full branch merge | ledger/source audit; runtime compatibility remains open | adapted |
| 1 | `108b04f` | Tier-1 prompt JSON serialization | `agents/paradigm_agent._build_structure_prompt`, `tests/test_paradigm_prompt.py` | PostgreSQL datetime-shaped values serialize through the prompt boundary without changing SQLite behavior | included in 71-test policy lane | ported |
| 1 | `041dc01` | test database isolation contract | `tests/conftest.py`, `tests/test_test_db_isolation.py` | unit/adapted-legacy entry forcibly clears production DB URL; disposable PostgreSQL is explicit | isolation test included in 71-test policy lane; source test file not cherry-picked | adapted |
| 1 | `789dd49` | CSS type-scale tokens only | `web/static/css/style.css`, current-format `CHANGELOG.md` entry | matching current-candidate literals use tokens; unrelated frontend bulk is excluded | source/CSS review; no UI acceptance claim | adapted |
| 1 | `6048a95` | `paper_orchestra_pipeline._complete_known_main_results_rows` | removed | layout code cannot invent benchmark numbers | static audit passed; isolated fixture pending | needs_ci |
| 1 | `6048a95` | `_deemphasize_significance_caveats` and callers | removed | editing cannot weaken uncertainty or claim limitations | static audit passed; isolated fixture pending | needs_ci |
| 1 | `6048a95` | `paperorchestra.full_pipeline` CRPP fallback | generic fail-closed boundary plus example plugin | generic manuscript path has no CRPP narrative or method aliases | AST/topic scan passed; runtime pending | needs_ci |
| 1 | `6048a95` | CGGR runners, auditors, shard tools, tests and docs | `plugins/examples/cggr` | domain code is opt-in and absent from default registry | registry/topic scan passed | adapted |
| 1 | `6048a95` | topic validation defaults and method aliases | explicit example plugin configuration | no topic method is a universal default | TOML/source text check passed | adapted |
| 1 | `9d24d29` | M1/M4 evidence rules | `contracts/scientific_evidence.py`, integrity fixtures | p=1/missing p/refuted/zero baseline/incomplete benchmark never confirm | 71-test policy lane passed; PostgreSQL/legacy runtime remains open | adapted |
| 2 | `7d0b42a` | `contracts/agenda.py` | hardened same path | mandatory positive token cap; zero GPU means disabled | AST passed; contract CI pending | needs_ci; rejected unlimited behavior |
| 2 | `7d0b42a` | `agents/direction_intake.py` | hardened same path | deterministic direction mapping and auditable echo | tests written, not run | needs_ci |
| 2 | `7d0b42a` | loader/selector/orchestrator/relevance | scoped modules plus persistence boundary | no implicit agenda and no untagged backlog fallback | source/SQL scan passed; PostgreSQL pending | needs_ci |
| 2 | `7d0b42a` | Agenda token ledger/budget | agenda ledger plus grant sub-reservations | reserve-before-call, pause/resume, hard cap, expiry recovery | disposable PostgreSQL multi-agenda/cap/fault harness passed; real backup/provider pending | adapted_needs_ci |
| 2 | `7d0b42a` | Agenda evidence gate | `agents/evidence_gate.py` plus canonical state machine | Agenda relevance cannot substitute for scientific evidence | tests written, not run | needs_ci |
| 2 | `7d0b42a` + `6048a95` | Agenda schema and missing problem/benchmark tables | `0001_meta_harness_v1.sql` | add-only, checksum journal, rerunnable, old backlog remains unscoped | physical backup restore with pgvector 0.8.1 passed first/second migration, all 48 count preservation, integrity and 6/6 repository integration; provider canary remains | adapted_needs_ci |
| 2 | `7d0b42a` | agenda web routes | operator-authenticated minimal API | run/observe mutations only; no merged legacy UI | static review only | needs_ci |
| 3 | `7d0b42a` | multi-provider LLM routing | `meta_harness/llm_routing.py`, granted role entry point | role separation, route provenance, cooldown/retry, fail closed | synthetic policy/fault passed; provider cooldown restart pending | adapted_needs_ci |
| 3 | both | legacy high-cost `call_llm` callers | proposal/ingestion/forge/validation/manuscript role adapters | stable identity plus scoped ResourceGrant; proposer/evaluator/reviewer provenance and independence | ingestion direct calls removed; 14 isolated legacy sites and CI remain | in_progress |
| 3 | production behavior | provider cooldown | `llm_provider_cooldowns` plus repository-backed router | auth/transient cooldown survives router/process reconstruction and cannot silently disappear | test written; PostgreSQL restart pending | needs_ci |
| 4 | `7d0b42a` | Colab lifecycle and account isolation | `meta_harness/compute.py`, `backends/colab_cli.py` | HOME/OAuth/session/quota isolation; secret refs; truthful failure | static only; CLI syntax/canary pending | needs_ci |
| 4a | `7d0b42a` behavior + new control plane | durable Colab admission/worker/recovery | `backends/colab_durable.py`, `orchestrator/colab_worker.py`, `colab_work_requests_v1` | persist request and compute claim before session; deterministic identity; measured settlement; lost remote control quarantines usage | disposable PostgreSQL queue/restart quarantine passed; real CLI/hardware canary pending | adapted_needs_ci |
| 4a | `7d0b42a` behavior | CPU validation admission/recovery | `orchestrator/meta_compute_runtime.py`, `orchestrator/auto_research.py` | durable grant/idempotency claim before execution; measured usage/artifact settlement; no false success | final targeted runtime regression plus physical PostgreSQL integration passed; actual CPU canary remains | adapted_needs_ci |
| 4a | application startup behavior | compute/auto-research startup order | `main.py`, `orchestrator/gpu_scheduler.py` | durable reconciliation finishes before auto-research may claim or submit work | test written; application/restart CI pending | adapted_needs_ci |
| 4 | production behavior + new contract | compute job lifecycle/idempotency | `meta_harness/compute_repository.py`, `compute_jobs_v1` | claim before transport, no duplicate restart submission, unknown outcome quarantined, artifacts/usage before success | disposable PostgreSQL compute/restart/fault lane passed; external backend pending | adapted_needs_ci |
| 4 | `6048a95` | SSH/GPU schedulers | durable scheduler admission plus guarded legacy transport | ResourceGrant on GPU queues; PostgreSQL direct queue insertion requires persisted compute identity; terminal failures require measured usage; recovery runs only under scheduler lock | legacy execution branches and runtime CI remain | in_progress |
| 5 | `6048a95` | benchmark design/protocol/manager/audit | retained benchmark modules plus `evidence_state.py` | exact monotonic evidence state machine | state tests written, held-out pending | needs_ci |
| 5 | both | legacy validator/status paths | canonical transition repository and scientific authority | operational result is `supported`; positive problem/knowledge/manuscript/meta-learning use requires persisted supported decision | selected core paths adapted; exhaustive runtime CI open | in_progress |
| 5a | validation-loop fault review | `agents/benchmark_audit.py`, `agents/validation_loop.py` | fail-closed fairness and pre-benchmark guards | candidate-only scoring, broad-context prompt propagation and zero-budget answer-shape changes are blocked; operational manifest verdict is `supported`, not scientific `confirmed` | validation-loop subset 22/22; synthetic fault lane 60/60; legacy classification recorded | adapted; PostgreSQL queue/evidence passed |
| 6 | `6048a95` | problem-first, novelty, idea taste | Frontier/portfolio feature inputs | features cannot allocate resources | contracts/policy written, not run | needs_ci |
| 6 | `9d24d29` | topic gate/surprisal | example feature input only | Frontier/portfolio owns allocation | topic-gate authority removal statically audited; runtime regression remains open | adapted |
| 6 | new contract | Frontier/Decision/Grant/Outcome | evidence-graph source, contracts, trusted assembler, repository, migration | evidence arrays come from scoped graph; auditable estimates, grants and actual metered outcomes | static/policy and disposable PostgreSQL evidence lane passed; API/real outcome pending | adapted_needs_ci |
| 7 | new contract | harness evolution objects | evolution/workspace/repository/evaluator modules, migration | isolated worktree/DB; hash-pinned read-only evaluator/holdout/candidate mounts; no network; held-in/out/canary plus approval | real disposable held-in/out/canary and negative isolation lanes plus final static revalidation passed; reviewer approval pending | adapted_needs_ci |
| 8 | `6048a95` | `deepgraph.toml` loader | additive policy/route/backend sections and minimal API | credentials are references; hard caps and trace roots configured | TOML/AST only; app not started | needs_ci |
| 8 | ingestion control gap | scoped ingestion operator API and durable worker | `ingestion_queue.py`, `scoped_ingestion_worker.py`, `scoped_ingestion_jobs_v1` | existing paper IDs only; exact active LLM grant; leased checkpoint resume; bounded retry; agenda-scoped mutation | disposable PostgreSQL enqueue/lease/checkpoint/retry/failure lane passed; provider/API pending | adapted_needs_ci |
| all | GitHub legacy | agenda-owned UPDATE/DELETE paths | `scripts/meta_harness_scope_audit.py` and scoped mutations/joins | every literal mutation carries explicit agenda scope; no cross-agenda legacy write | static audit clean; PostgreSQL fault CI pending | needs_ci |

The 30 adapted-legacy failures are individually classified in
[LEGACY_TEST_CLASSIFICATION.md](LEGACY_TEST_CLASSIFICATION.md). No old
grantless, unscoped, password-bearing, topic-default or unlimited behavior is
restored for compatibility.

## Deliberately not ported in v1

- Both complete web UIs and production dashboard assets.
- PaperBanana, image assets, pixel office, historical papers/submissions.
- Venue and manuscript template bulk.
- Topic-specific output values, method aliases, ablations, runners, and
  manuscript prose in the generic runtime.
- Production deployment helpers and any automatic master replacement.
- A trained resource policy model; v1 uses a transparent logged heuristic.
