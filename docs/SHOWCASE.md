# Showcase: numbers, demo route, cases

> 中文导读: 本页集中展示 DeepGraph 的运行规模、现场演示路线和代表性案例。
> 所有数字均标注采集日期; 规模数字属于运行数据。

## Scale (operational snapshot, 2026-08-06)

Collected live from `/api/stats` on the running deployment (port 8080):

- Papers: 21,151 collected, 6,729 fully processed
- Structured evidence: 28,700 claims, 156,194 results, 238 contradiction
  clusters detected
- Knowledge graph: 248,441 entities, 735,913 relations, 5,051 taxonomy nodes,
  32,252 paper-taxonomy assignments
- Discovery: 11,936 insights, 110 deep insights (39 paradigm-level, 71
  paper-idea-level), 25,652 mapped research opportunities
- Execution: 113 experiment runs, 99 GPU jobs across 17 registered workers
- Investment: 921M LLM tokens

These numbers grow continuously while ingestion and discovery run.

## Demo route (15 minutes)

Open the dashboard (`http://<host>:8080`) and walk the loop end to end:

1. **Overview** -- papers, graph, insights, runs and token totals at a glance.
2. **Explore Map** -- pick an ML subfield node: node summary, its papers, and
   generated insights side by side.
3. **Evidence** -- the method x dataset matrix; strong cells, weak cells, and
   white space (research opportunity) are immediately visible.
4. **Paper Ideas** -- Tier 1 / Tier 2 deep insights with their gate status and
   experiment linkage.
5. **Experiments** -- runs with their two status badges (operational and
   scientific), artifacts, and verdicts.
6. **Manuscripts** -- generated PDFs/TeX, quality reports, and bundle status.
7. **Agent Office** -- the big agents (extraction, graph, ideas, planning,
   execution, manuscripts, orchestration) and what each is doing right now.

## Case: the accounting loop, closed end to end

On 2026-08-04 the full authorization -> execution -> settlement chain ran
autonomously for the first time: a candidate passed the topic gate, received a
scoped resource grant, executed, was settled token-by-token, and the unused
reservation was refunded to its agenda budget automatically -- with every step
leaving an auditable record (`OutcomeRecord` id=1). The pipeline that spends
resources and the ledger that accounts for them agree by construction, not by
convention.

## Case: a paper-grade benchmark harness (CGGR plugin)

`plugins/examples/cggr/` is a complete, self-contained experiment harness
showing what "contracted experiment" means in practice:

- A frozen benchmark contract pins datasets (MuSiQue-Ans, StrategyQA,
  2WikiMultihopQA and a counterfactual stress split), a pinned Qwen instruct
  checkpoint, and seeds `[0-4]`.
- The runner emits machine-readable artifacts for every run: per-seed results,
  ablation tables, `failure_cases.jsonl`, and raw generations
  (`plugins/examples/cggr/experiment_runner.py`).
- Statistics are computed, not asserted: paired permutation tests and
  2,000-round bootstrap confidence intervals ship with the harness.

## Case: a system that can say no

Manuscript generation is gated, and the gates are load-bearing:

- A claim without quantitative evidence cannot appear in the abstract,
  introduction, or conclusion (`agents/paper_completeness.py`).
- Fewer than three seeds, a missing baseline, or an absent significance test
  blocks the evidence audit (`contracts/scientific_evidence.py`).
- A bundle that fails its quality gates is stamped `DO_NOT_SUBMIT.md` with a
  concrete repair list instead of being shipped
  (`orchestrator/manuscript_watchdog.py`).

For a research-automation system, knowing when *not* to publish is the
feature that makes everything else trustworthy.
