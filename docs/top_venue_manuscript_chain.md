# Top-Venue Manuscript Chain

This note records the current DeepGraph paper-production standard and the
pipeline changes that enforce it. After issues #11-#15 the chain is no longer
ICLR-only: it routes each draft to the venue best matched to the insight and
the submission window, and runs adapter-specific format checks before any
bundle reaches the gate.

## Reference Corpus

- Local reference papers live in `H:\Deepgraph\workspace\pdfs`.
- The current corpus contains 292 PDF files, including `2604.14206.pdf`,
  `2604.15356.pdf`, and related 2604-series papers. One observed zero-byte
  file, `2604.22554.pdf`, is skipped by profiling, leaving 291 non-empty PDFs.
- Failed text extraction through MiKTeX `pdftotext` should not block the
  pipeline. The bundle quality report now runs a PyMuPDF-based
  `reference_corpus_audit_v1` profile against this corpus and adds explicit
  issues when a generated paper is too short, has sparse citations/figures, is
  missing expected section signals, or lacks the problem-motivation-method-result
  spine.

## Existing DeepGraph Draft Pattern

- Generated DeepGraph research artifacts include CGGR reports under
  `deploy_bundle\artifacts\research\deep_research_di_13_*` and related
  `deep_research_di_17_*` directories.
- The useful structure in those reports is the problem-first CGGR framing:
  selective deliberation is a decision problem over marginal utility, not a
  generic chain-of-thought prompt variant.
- The weak pattern to avoid is artifact reporting without a crisp paper spine:
  question, motivation, mechanism, result, and falsification.

## Venue Routing Architecture (issues #11-#15)

```
paper_idea_agent
       │
       ▼
venue_router.py ──reads── agents/venues.yaml  (6 venues registered)
       │                  ├─ iclr2026     (single_column, iclr2026_conference)
       │                  ├─ neurips2025  (single_column, neurips_2025)
       │                  ├─ acl_arr      (two_column,    acl_natbib)
       │                  ├─ emnlp2025    (two_column,    acl_natbib)
       │                  ├─ cvpr2024     (two_column,    ieee_fullname)
       │                  └─ iccv2025     (two_column,    ieee_fullname)
       │
       ▼   (primary, secondary, reasons)
manuscript_templates.get_adapter(template_id) → TemplateAdapter
       │
       │   .column_layout / .bibstyle_name / .max_pages / .required_packages
       │   .normalize_source(body) / .submission_mode toggle
       ▼
format_linter.lint_manuscript(source, adapter, page_count=...)
       │   12 checks total (rule_set = format_linter_v1):
       │     7 structural   — documentclass, bibstyle_matches_venue,
       │                       required_packages, page_count, figure_placement,
       │                       column_layout_consistency, figure_grid_density
       │     5 issue-#14    — font_size_consistency, section_spacing,
       │                       float_density, citation_density, bib_style_match
       │   persist_lint_run(...) writes to `format_lint_runs`
       ▼
paper_orchestra_pipeline.require_submission_ready()
       │   blocks synthetic data, missing problem_awareness contract,
       │   or any lint check with severity=error
       ▼
manuscript bundle  (per-venue .sty + .bst copied from `third_party\<venue>`)
       │
       ▼
web/manuscript_routes.py
       /api/manuscript/route        — router suggestion JSON
       /api/manuscript/lint         — on-demand lint
       /api/manuscript/bundles      — list per-venue bundles
       dashboard panel "Manuscript Routing" — human review
```

The router is a pure function of `(insight_metadata, now, available_venues)`;
its scoring weights live in `venues.yaml` so adding a venue is config-only —
drop a new entry plus the matching `manuscript_templates/<id>.py` adapter and
the rest of the chain picks it up.

## Enforced Standard

- Submission bundles use the official template files for the routed venue,
  copied from `third_party\<venue_id>\<venue_id>`. ICLR 2026 is still the
  fallback when the router has no signal, but it is no longer hard-coded.
- Manuscript state carries a `problem_awareness` contract with:
  `central_question`, `motivation`, `method_answer`, `result_claim`, and
  `falsification_result`.
- Evidence gates block full-paper generation when the problem-awareness
  contract is missing or when `format_linter` returns any `severity=error`
  finding (e.g. wrong bibstyle, page-count overage > 1).
- The five issue-#14 checks each have a fail fixture in
  `tests/test_format_linter.py::test_dirty_fixture_triggers_all_five_issue14_checks`
  and are referenced by name in `artifacts/d3_format_linter_acceptance.json`.
- API/PaperBanana conceptual figures are deferred until after an initial
  manuscript draft exists. Early plotting remains native and evidence-backed.
  `scripts/demo_full_paper_compile.py` now generates real PPL/wallclock/ablation
  curves with matplotlib so the 10/10 compiled PDFs carry non-empty figures.
- GPU execution must use the locked benchmark contract from each run spec. The
  launcher records model/dataset/example/seed env vars in execution reports so
  paper evidence cannot silently drift from the benchmark manifest.
- Stale SSH GPU jobs are recovered periodically by the scheduler, not only when
  the scheduler starts. Runs that crash after model load are surfaced with a
  failure type and the last `BENCHMARK_STAGE` marker.

## Chain Ownership

- `paper_idea_agent.py`: produces problem-aware paper ideas.
- `agents/venue_router.py` (#12): selects primary/secondary venue from
  `agents/venues.yaml`.
- `agents/manuscript_templates/` (#13): per-venue `TemplateAdapter`
  implementations (column layout, bibstyle, page budget, packages,
  `normalize_source`, `submission_mode` toggle).
- `agents/format_linter.py` (#14): 12-check linter with persistence and
  LLM tiebreaker hook.
- `web/manuscript_routes.py` (#15): API + dashboard surface for routing and
  lint runs.
- `experiment_forge.py`: freezes the experiment evidence contract and the
  problem-awareness contract before execution.
- `result_interpreter.py`: carries problem awareness from experiment outcomes
  into result packets.
- `manuscript_pipeline.py`: builds the canonical manuscript input state.
- `paper_orchestra_prompts.py` and `paperorchestra/full_pipeline.py`: enforce
  the problem-motivation-method-result spine during writing.
- `paperorchestra/figure_orchestra.py`: separates early data figures from
  post-writing API diagrams.
- `paper_orchestra_pipeline.py`: assembles per-venue bundles, calls
  `require_submission_ready()`, and records quality reports against the local
  PDF corpus, including `reference_corpus_audit_v1`.
- `reference_corpus_audit.py`: profiles local reference PDFs and generated TeX
  manuscripts so manuscript quality gates compare drafts to the real corpus
  rather than only to generic heuristics.
- `ssh_gpu_backend.py`, `validation_loop.py`, and `gpu_scheduler.py`: keep GPU
  execution aligned with the benchmark contract and recover stale remote runs.

## Evidence Trail

| Issue | Evidence JSON | Notes |
|-------|---------------|-------|
| #11 (epic) | `artifacts/manuscript_venue_routing_acceptance.json` | umbrella, references d1-d4 |
| #12 (D1)   | `artifacts/d1_template_router_acceptance.json` | router + adapter base + venues.yaml |
| #13 (D2)   | `artifacts/d2_top_venue_adapters_acceptance.json` | 6 venues × 3 fixtures, sha256 distinct |
| #14 (D3)   | `artifacts/d3_format_linter_acceptance.json` | 12 checks, db roundtrip True |
| #15 (D4)   | `artifacts/d4_manuscript_routing_api_acceptance.json` | API contract + dashboard hooks |
| #9         | `artifacts/agenda_loop_acceptance.json` | agenda-driven loop + revision plan |
