# CGGR Top-Venue Novelty and Baseline Gap

This is a live positioning note for the CGGR paper. It is not a claim of novelty by itself.

## Current Novelty Position

The exact project term `Counterfactual Gain Gated Reasoning` is not the paper claim. The defensible claim is narrower:

- CGGR frames selective deliberation as an answer-now versus deliberate-more decision.
- The gate estimates the counterfactual utility gain of extra reasoning relative to the immediate-answer branch.
- Routing is conservative: extra reasoning is used only when a lower confidence bound on estimated gain is positive.
- The active executable runner is a fixed proxy-gated instantiation, not a trained estimator: it maps a task-aware difficulty/question-structure proxy to short versus deliberative token budgets and reports the resulting routing traces for audit.
- The evaluation is artifact-gated and includes raw predictions, routing traces, cost/latency, ablations, uncertainty, and simple-case degradation checks.

## Nearby Prior Art That Must Be Acknowledged

| Prior-art area | Representative source | Why it matters |
| --- | --- | --- |
| Rational metareasoning / value of computation | `Rational Metareasoning for Large Language Models`, OpenReview: https://openreview.net/forum?id=jRZ1ZeenZ6 | Already casts LLM reasoning control as deciding when computation is worth its cost. CGGR must not claim the broad value-of-computation framing as new. |
| Certainty-based adaptive reasoning | `Prolonged Reasoning Is Not All You Need: Certainty-Based Adaptive Routing for Efficient LLM/MLLM Reasoning`, arXiv: https://arxiv.org/abs/2505.15154 | Already switches between short and long reasoning from uncertainty/certainty signals. CGGR must distinguish counterfactual gain from certainty gating. |
| Self-routing between response modes | `Self-Route: A Training-Free Method to Cost-Efficiently Route Inputs Between LLMs and Reasoning LLMs`, arXiv: https://arxiv.org/abs/2505.20664 | Already routes between general and reasoning modes. CGGR must avoid claiming that selective reasoning-mode routing is itself novel. |
| Joint model/strategy routing | `Route to Reason: Adaptive Routing for LLM and Reasoning Strategy Selection`, arXiv: https://arxiv.org/abs/2505.19435 | Already learns an adaptive router over both LMs and reasoning strategies under budget constraints. CGGR must acknowledge this as a close routing baseline and cannot claim broad superiority without a registered comparable artifact. |
| Cost-quality model routing | `RouteLLM: Learning to Route LLMs with Preference Data`, arXiv: https://arxiv.org/abs/2406.18665 | Establishes cost-quality routing as a strong adjacent baseline. CGGR should be positioned as within-input reasoning-budget routing, not generic model routing. |

## Claims Currently Allowed Before Extra Baselines

- CGGR is a counterfactual-gain formulation of selective deliberation.
- The current benchmark tests CGGR against the locked deployable baselines in `run_45` and `run_46`.
- If `run_47` passes full audit, the paper may report CGGR versus those locked baselines with uncertainty, cost, ablation, and routing evidence.

## Claims Blocked For Top-Venue Submission

- Do not claim CGGR is the first adaptive reasoning or routing method.
- Do not claim state of the art unless the contract includes current adaptive reasoning baselines and their audit passes.
- Do not claim superiority over CAR, Self-Route, Route-to-Reason, rational-metareasoning, or model-routing approaches from the current run alone.
- Do not infer broad model-family generality from the single-model Qwen2.5-14B contract.

## Top-Venue Strengthening Checklist

- Add directly comparable adaptive-reasoning baselines where feasible: CAR-style certainty routing, Self-Route-style mode routing, and a rational-metareasoning/value-of-computation policy.
  - Runner support is now implemented behind `DEEPGRAPH_BENCHMARK_INCLUDE_TOP_VENUE_BASELINES=1`.
  - Exact method labels: `CAR-Style Certainty Adaptive Routing`, `Self-Route-Style Mode Routing`, `Rational-Metareasoning VOC Routing`.
  - Shard preparation support: `scripts/prepare_cggr_top_venue_baseline_shard.py`; the prepared local template is `workspace/tmp/cggr_top_venue_baseline_shard_template`.
  - The shard preparation script sanitizes stale learned-router wording: generated top-venue baseline plans must state that the active executable evidence is fixed proxy-gated, with learned-estimator/router claims disabled unless a future artifact explicitly trains and audits one. It also filters oracle upper-bound diagnostics out of deployable baseline requirements.
  - Audit support is now available through `scripts/audit_paper_benchmark_artifacts.py --require-full --require-top-venue-baselines`.
- Add a non-deployable oracle routing diagnostic only if pre-registered and clearly separated from deployable methods.
- Report cost-quality Pareto curves, not only aggregate utility.
- Acknowledge Route-to-Reason-style joint model/strategy routing; direct superiority over it requires a registered comparable implementation or an explicit limitation that the current contract does not compare against it.
- Add cross-model validation or clearly scope the result to Qwen2.5-14B-Instruct.
- Preserve the current audit discipline: every new baseline must produce raw predictions, routing traces, cost/latency, failure rows, and paired statistics.

Until these items are satisfied, the honest target is: a strong, audited selective-deliberation paper with a distinctive counterfactual-gain framing, not a broad state-of-the-art adaptive-reasoning claim.
