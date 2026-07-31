# Changelog

This file records user- and operator-visible DeepGraph changes. The format is
inspired by Keep a Changelog, but an entry is not considered released until it
has an immutable commit, isolated acceptance evidence, explicit approval and a
tag/deployment record.

## [Unreleased] — meta-harness-v1 integration candidate — 2026-07-31

Status: **development branch; not accepted for master; not deployed**.

Local implementation checkpoint:
`c25e63c` (`feat: build controlled meta-harness-v1 candidate`). This hash has
not been pushed and is not an acceptance/deployment identifier.

### Lineage and why this candidate exists

The candidate follows the controlled B integration:

```text
GitHub origin/master@6048a95
  -> local integration/meta-harness-v1
  -> scientific-integrity/topic-boundary cleanup
  -> semantic ports from production/topic references
  -> isolated PostgreSQL + candidate worktree + canary acceptance
  -> master replacement only after explicit approval
```

- Base: `6048a9568c79b011074e0dba2662fd473cfab250`
  (`Document paper quality roadmap`, 2026-06-26). This GitHub line supplies the
  newer problem-first, idea-taste, benchmark, dataset, scout, feedback and
  loop-router architecture.
- Production behavior reference:
  `7d0b42af8e8f061c3c16800c44224c110f3b94a0`
  (`本地 6/21 快照:多账号GPU池/Colab后端/verdict闸等未提交改动`,
  2026-07-01).
- Topic-gate/integrity reference:
  `9d24d29c6a7d1017301ffa9c36ff9b4b3dfae88d`
  (`选题三问闸门 + 惊讶度驱动算力分配,并解禁诚实负结果`,
  2026-07-29).
- Production and topic-gate are sibling commits with parent
  `4f78f828704567f4210b8628973d4a0e6ba62868`. That lineage contains historical
  v0.2/v0.3 changelog entries, including the earlier Agenda implementation.
- GitHub master and the production/topic lineage have no merge-base. Therefore
  this work does not merge the histories and does not cherry-pick the large
  production/topic commits. Capabilities are ported by behavior contract.
- Production `7d0b42a` remains unchanged as behavior/migration/rollback
  reference. It is not the candidate base and is not rewritten by this branch.

### Added

- Versioned Agenda contract, direction intake, loader, relevance, selector and
  fair multi-agenda queue.
- Positive token hard caps, GPU-hour caps, concurrency limits, backend
  allowlists, reserve/settle/release ledger and budget pause/resume.
- Explicit audited legacy-backlog import; old unscoped backlog remains excluded.
- Additive PostgreSQL migration with checksum journal and a guarded,
  dry-run-by-default runner.
- Additive creation of GitHub-line `research_problems`,
  `experimental_evidence_edges` and `benchmark_harness_jobs` for production
  restores where they are absent.
- `FrontierPacket`, `IdeaDecisionPacket`, `ResourceGrant` and `OutcomeRecord`
  contracts with provenance, confidence intervals and agenda scope.
- Frontier freshness/obsolescence gate and transparent best-of-N portfolio
  policy with diversity, exploration, falsification, surprise and opportunity
  cost.
- Role-separated proposer/evaluator/reviewer LLM routing, explicit fallback,
  cooldown, metering and failure observations.
- Durable provider cooldown state that is reloaded after router/process
  reconstruction instead of existing only in one call object.
- Backend-neutral CPU/LocalGPU/SSHGPU/ColabGPU contracts with capability,
  submit, heartbeat, cancel, timeout, artifact and usage interfaces.
- PostgreSQL-backed compute claim/idempotency lifecycle: claim before backend
  submission, quarantine uncertain outcomes, reuse a persisted live job after
  restart, and require bounded usage plus grant-required artifacts before
  durable success.
- Multi-account Colab lifecycle adapter with isolated HOME/OAuth/session/quota,
  secret references and credential/backup path rejection.
- Single monotonic scientific evidence state machine from `planned` through
  `manuscript_allowed`.
- Content-addressed evidence-audit and scientific-decision records that bind
  raw artifacts, claim ledger, benchmark, evaluator, holdout and verdict.
- HarnessCandidate/Patch/FailureCluster/EvaluationRun/RegressionReport/Archive,
  candidate worktree/environment isolation and agenda-scoped persistence.
- Outcome calibration report using success Brier score, token/GPU MAE and
  impact RMSE; policy changes remain reviewer-controlled.
- Trusted OutcomeRecord assembly from persisted grant metering, compute usage,
  experiment artifacts and canonical scientific decisions; the operator API
  no longer accepts caller-supplied outcome metrics.
- Minimal operator-authenticated meta-harness API and count-only status view.
- Non-sensitive `deepgraph.toml` policy for agendas, portfolio, grants, LLM
  roles, compute backends, evidence, harness evolution, failures and traces.
- Static source audit, isolated PostgreSQL test skeleton, scientific-integrity
  fixtures, fault tests, runbooks and detailed acceptance checklist.

### Changed

- Auto-research default cycle now queues only explicitly agenda-scoped work for
  portfolio review; legacy global event/backlog consumption is not invoked.
- Core problem, idea, job, run, artifact, claim, manuscript and evidence-edge
  writes carry agenda scope.
- Experiment feedback updates agenda-local signal outcomes rather than shared
  ingestion counters, preventing cross-agenda learning leakage.
- GPU/forge/validation/benchmark core paths require a matching active,
  non-expired ResourceGrant.
- Forge scout/scaffold/repair and validation code/reproduction repair use the
  granted proposer route with stable idempotency keys and route provenance.
- Validation reports operational `supported` instead of directly creating
  `confirmed`; positive problem, knowledge, manuscript and meta-learning paths
  require a persisted supported scientific decision.
- GPU jobs fail when validation returns failed/blocked/invalid or no result;
  such failures are no longer persisted as completed jobs.
- Manuscript generation requires `manuscript_allowed` evidence plus reviewer
  approval.
- Validation and manuscript review apply the same scientific evidence rules:
  missing/non-significant p-values, refutation, zero/missing baselines, missing
  metrics and incomplete benchmarks cannot confirm a claim.
- SSH metadata stores credential references instead of passwords, strict host
  checking is enabled and implicit remote dependency installation defaults off.

### Removed or isolated from the generic runtime

- Removed fixed idea8/run13 VOC result completion.
- Removed automatic weakening of non-significance and uncertainty caveats.
- Removed CRPP-specific deterministic scientific prose from the generic
  PaperOrchestra path.
- Moved CGGR/CRPP/idea8 runners, ablations, aliases, shard tools, result
  auditor, tests and historical docs to `plugins/examples/cggr`.
- Removed topic scripts from the default generic agent registry.

The example plugin is non-production and disabled unless explicitly selected.

### Scientific and safety invariants

- `p=1` is not significant.
- A missing p-value cannot support a significance claim.
- `refuted` cannot produce a positive claim even with a low p-value.
- Zero/missing baseline, missing metric or incomplete full benchmark cannot
  confirm.
- Presentation/layout/refinement code cannot invent numbers or strengthen a
  scientific claim.
- Pilot evidence cannot become confirmed.
- Provider/backend failure cannot become completed/confirmed.
- No GPU or high-cost role-routed LLM request is authorized without a matching
  ResourceGrant.
- Candidate code cannot modify evaluator, held-out, canary, budget, migration,
  scientific evidence or safety policy paths.

### Deliberately not included

- Merging both complete Web UIs.
- PaperBanana/image assets, pixel office, historical generated paper trees and
  bulk venue templates.
- Automatic deployment, push or master replacement.
- A trained policy model.
- Production-host pytest, benchmark, GPU, SSH or Colab execution.

### Validation status

Allowed static checks currently report:

- 239 Python files parsed by AST;
- no finding from the topic/integrity/migration/secret static audit;
- SQL AST audit: 757 literal calls, 754 statically countable, no definite
  mismatch, and 109 dynamic calls left for review/CI;
- additive migration plan: 81 statements, 23,995 bytes, SHA-256
  `dcdf8fcce3113a36f8c652b5f015135921b8541c68523e6b01cb576e0c8aecb9`,
  no destructive token and no database access;
- `git diff --check` passed;
- agenda example JSON parsed.

Not run: pytest, application imports/startup, PostgreSQL migration, production
backup startup, provider/backend calls, CPU/GPU/SSH/Colab work, candidate
worktree evaluation, held-in/held-out/canary, restart recovery and rollback
rehearsal. The host Python 3.9 has no TOML parser, so TOML runtime parsing also
remains an isolated CI item.

### Known incomplete integration

- Some post-agenda legacy LLM callers still use the legacy client instead of
  the granted role route, notably pre-idea discovery and parts of manuscript
  refinement. A pre-candidate grant must not be faked with the wrong idea ID.
- The legacy GPU scheduler still contains transport-specific branches and is
  not yet wired to the new durable `ComputeJobRepository` at runtime.
- Not every legacy direct operational/scientific status write has been routed
  through the canonical evidence transition repository.
- Live Frontier retrieval and trusted automatic OutcomeRecord assembly are not
  end-to-end runtime wired; successful outcomes have trusted assembly code,
  while failed compute jobs still need durable usage capture before they can
  settle an OutcomeRecord.
- Reviewer identity/approval needs an externally authenticated mechanism.
- No isolated CI/canary evidence exists yet, so this entry must remain
  Unreleased and the branch must not replace master.

Detailed status:
[implementation checklist](docs/integration/IMPLEMENTATION_CHECKLIST.md),
[LLM caller inventory](docs/integration/LLM_CALLER_INVENTORY.md),
[porting ledger](docs/integration/PORTING_LEDGER.md), and
[master acceptance matrix](docs/integration/ACCEPTANCE_EVIDENCE.md).
