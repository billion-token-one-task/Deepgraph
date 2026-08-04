# Claude Code task: README overhaul - truth, structure, and innovations

Read and follow this entire file before acting. This is a documentation task
scoped to the repository only.

## Objective

Rewrite `README.md` so the repository's front door truthfully reflects the
system as of 2026-08-04, after: the meta-harness v1 upgrade, the frontend
merge (provenance UI, two-register badges, i18n, direction submission), the
recovery workstream (topic gate, Frontier bootstrap authority, ResourceGrant
economics, selfheal watchdog), and the first closed authorization-execution-
settlement chain. Bold cuts of stale content are wanted, substantial new
content where it matters is wanted; the only hard rule is that every sentence
must be currently true and verifiable.

Specifically requested by the product owner:

1. prune outdated and duplicated material aggressively;
2. expand the parts that matter, even if total length grows;
3. add a dedicated section presenting the system's innovations; and
4. fix the changelog situation (see "Changelog decision" below).

## Truthfulness rules (non-negotiable)

- The system has NO scientific findings yet: `scientific_decisions_total` is 0.
  The first pilot executed and was honestly settled as failed (OutcomeRecord
  id=1: 2612 tokens spent, execution_result=failed, verdict=inconclusive,
  ladder correctly stayed at planned, unused reservation refunded). You may
  present this as evidence the accounting is honest - never as a result.
- Operational counts (6,729 papers analyzed, 97 ideas, 111 experiment runs as
  of 2026-08-04) are inputs and process, not achievements. Keep the
  two-register discipline everywhere: operational status and scientific
  status are different things and the README must never blur them.
- Date every statistic. Prefer mechanisms over numbers.
- Do not oversell autonomy: the global autonomy switches are currently OFF;
  the system runs supervised, gated, budgeted flows.

## Innovations section - seed list (verify each in code before writing)

Present these as designed mechanisms with file references, not marketing:

- Evidence ladder with content-hash gates: planned -> sanity_passed ->
  full_benchmark_complete -> evidence_audited -> scientifically_decided ->
  manuscript_allowed; every transition audited, hashes from real artifacts,
  reviewer signature required for manuscript_allowed
  (`contracts/meta_harness.py`, `meta_harness/evidence_state.py`,
  `meta_harness/repository.py`).
- Two-register status model carried to the UI: RUN badge (operational) vs
  EVIDENCE/DECIDED badge (scientific), never merged; failures and refused
  gate transitions are first-class timeline events
  (`web/provenance_routes.py`, `web/static/js/app.js`).
- Topic gate: admission measured in bits of surprise with a written
  prediction; no LLM inside the gate; no enable/disable bypass switch
  (`agents/topic_gate.py`, `meta_harness/topic_gate_admission.py`).
- Frontier bootstrap authority: a deliberately smaller-than-grant, bounded,
  TTL-limited, single-purpose authority that resolves the
  grant-needs-evaluator-needs-grant circularity and fails closed on every
  path (`meta_harness/frontier_authority.py`).
- ResourceGrant economics: per-stage token/GPU caps, TTL expiry, append-only
  reservation ledgers, exact settlement including of failures (the first
  OutcomeRecord is the worked example).
- Retrospective review: a human-signed (HMAC), verdict-capped, fully audited
  path that lets pre-governance history onto the ladder without diluting it
  (`meta_harness/retrospective_review.py`).
- Direction intake: a research direction submitted as a small config becomes
  a budgeted, scoped agenda (`agents/direction_intake.py`,
  `research_agendas/harness_edit_loop_study.v1.json` as the worked example).
- Operations discipline: manifest-driven SHA-verified deployments with
  diff-gates and rollback anchors (`deploy/manifest/`,
  `scripts/build_deployment_manifest.py`), a self-heal watchdog that holds
  with reason codes and detects DB idle-in-transaction stalls, and a
  self-destructing post-deploy cleanup timer.

Cut candidates (verify then remove): the embedded `## Changelog` section
(stale since 2026-06-26, duplicates `CHANGELOG.md`), any stale `## Status`
claims, architecture text describing retired paths (the unscoped paper
worker is retired; ingestion is grant-scoped now), anything describing the
pre-merge frontend.

## Changelog decision (ask the product owner before touching CHANGELOG.md)

A 2026-06-10 decision recorded that `CHANGELOG.md` is partner-maintained and
must not be edited locally. Circumstances have changed (this repo's master is
now pushed directly by the owner's sessions). Ask the owner ONE question
before editing `CHANGELOG.md`: keep the old convention (README links to
CHANGELOG.md, embedded README changelog section is deleted) or take over
CHANGELOG.md maintenance now. Default if no preference: delete the embedded
README section, link to CHANGELOG.md, leave CHANGELOG.md itself untouched.

## Required reading before writing

- `README.md` and `CHANGELOG.md` as they are;
- `docs/frontend/FRONTEND_DISCOVERY_AND_RECOMMENDATION.md` and
  `FRONTEND_MERGE_IMPLEMENTATION.md`;
- `docs/runbooks/RECOVERY_2026-08-03.md`;
- project memory entries (auto-loaded): deepgraph-ops-rules,
  deepgraph-frontend-merge-v1, deepgraph-meta-harness-chain-closed,
  deepgraph-gpu-fleet.

## Boundaries

- Repository files only. Do not touch the live service, its database, .env,
  systemd, or anything under /home/billion-token.
- Shared worktree: other sessions work in this checkout. Run `git status`
  before committing and commit ONLY the files you changed. Never sweep other
  sessions' files into your commits.
- English stays the primary README language; a short Chinese orientation
  paragraph near the top is welcome but optional.
- Do not push; the owner pushes. Stop after committing and present a summary
  of what changed and why.

## Deliverable

A rewritten `README.md` (and `CHANGELOG.md` only if the owner approves the
convention change), committed on master with a clear message, plus a short
before/after summary: sections removed, sections added, every claim's
verification source. Stop and wait for the owner's review.
