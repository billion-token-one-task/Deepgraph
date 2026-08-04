# Claude Code task: drive agenda 7 (harness-edit-loop-study-v1) supervised

Read this whole file, then the referenced code and logs, before acting. You
are the research driver for ONE agenda. Global autonomy is OFF and stays off;
you advance the research through the sanctioned, supervised machinery only.

## Objective

Produce agenda 7's first real research output: pre-registered experiment
plans (its `required_output.goal` is `experiment_plan`), by driving the
meta-harness cycle end to end under supervision:

signals/frontier -> topic gate -> portfolio decision -> ResourceGrant ->
bounded, budgeted plan-production work -> honest settlement.

The topic (config: `research_agendas/harness_edit_loop_study.v1.json`):
on a FIXED small task suite with a FIXED base model, does a bounded
propose-validate-accept harness-edit loop (weakness mining -> single-surface
bounded edit -> held-in AND held-out non-regression gate) improve task pass
rate over a frozen harness, and which single editable surface wins. One
falsifiable question; single editable surface per experiment; written
prediction and pre-registered held-in/held-out split are mandatory in every
plan; negative results are valid outcomes.

## Required reading first (do not skip)

1. The worked example: the recovery workstream drove agenda 5 through the
   first full cycle on 2026-08-04. Read `/home/ec2-user/deepgraph-rollback/OPERATORS.log`
   and project memory entries `deepgraph-meta-harness-chain-closed`,
   `deepgraph-ops-rules`, `deepgraph-frontend-merge-v1`.
2. The machinery: `meta_harness/frontier_bootstrap.py`,
   `meta_harness/frontier_authority.py` (token ceiling <= 20k, TTL <= 120min,
   one pinned provider/model, fail-closed), `meta_harness/topic_gate_admission.py`,
   `agents/topic_gate.py`, `meta_harness/portfolio.py`,
   `orchestrator/bounded_execution.py` (execute_granted_candidate),
   `web/meta_harness_routes.py` (operator API surface: /frontier,
   /frontier/from-evidence-graph, /portfolio/decide, /grants,
   /ingestion/jobs; operator token in the live .env).
3. Design invariants you must not fight: a settled candidate (one with an
   OutcomeRecord) can never be re-run; grants attach only to candidates at
   `awaiting_portfolio_decision`; verdicts and gate refusals are recorded, not
   retried blindly.

## Suggested course (adapt to what the code actually requires)

1. Inventory signals: does the corpus carry the agenda's key literature
   (STOP arXiv:2310.02304, ADAS 2408.08435, AFlow 2410.10762, and recent
   self-harness work)? Check taxonomy ml.agents.* coverage. If thin, run ONE
   scoped ingestion job (<= 20 papers, via /ingestion/jobs with its own small
   grant) before the frontier step.
2. Create/persist a research problem for agenda 7 and produce a Frontier
   packet (bootstrap authority route or from-evidence-graph), gate it.
3. Portfolio decision for the admitted candidate; issue a pilot-stage grant
   (token_cap <= 5000 per grant; agenda 7 lifetime budget is 100,000 - treat
   it as scarce).
4. Bounded execution to produce the deliverable: for THIS agenda the
   deliverable of v1 is a pre-registered experiment plan (spec + prediction +
   split + acceptance rule), not an executed experiment. cpu+llm backends
   only; no GPU.
5. Settle honestly. A gate refusal or a failed step is a recorded result -
   report it, do not paper over it.
6. Iterate while budget and signal quality justify it; stop and report after
   the first 2-3 settled cycles either way.

## Boundaries

- Never touch: autonomy switches, .env, systemd, deployments, agenda 5 (the
  recovery workstream's lane), any other agenda's rows.
- All mutations via the operator-token APIs or the reviewed module entry
  points the worked example used. No hand-written UPDATEs to business tables.
- Append one line to /home/ec2-user/deepgraph-rollback/OPERATORS.log before
  and after each mutating step (who/what/why).
- Spend discipline: <= 5,000 tokens per grant, <= 25,000 total without
  fresh user approval. Frontier authority caps per contract.
- Shared worktree: git status before any commit; commit only files you
  created; do not push (the owner pushes).
- If the machinery refuses something, read why before retrying; a refusal
  with a reason code is the system working. Report refusals verbatim.

## Deliverable

A written cycle report: signals used, gate verdicts (with codes), decisions,
grants issued/settled with exact token accounting, and the produced
experiment plan(s) - plus where to see each step in the UI (Process timeline
and Ideas tab under agenda 7). If blocked, the report states the exact
blocker and what it needs. Stop for owner review after the report.
