# Top-Venue Manuscript Chain

This note records the current DeepGraph paper-production standard and the
pipeline changes that enforce it.

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

## Enforced Standard

- All default submission bundles use the official ICLR 2026 template files from
  `third_party\iclr2026\iclr2026`.
- Conference bundles must copy `iclr2026_conference.sty`,
  `iclr2026_conference.bst`, `math_commands.tex`, `natbib.sty`, and
  `fancyhdr.sty`.
- Manuscript state carries a `problem_awareness` contract with:
  `central_question`, `motivation`, `method_answer`, `result_claim`, and
  `falsification_result`.
- Evidence gates block full-paper generation when this problem-awareness
  contract is missing.
- API/PaperBanana conceptual figures are deferred until after an initial
  manuscript draft exists. Early plotting remains native and evidence-backed.
- GPU execution must use the locked benchmark contract from each run spec. The
  launcher records model/dataset/example/seed env vars in execution reports so
  paper evidence cannot silently drift from the benchmark manifest.
- Stale SSH GPU jobs are recovered periodically by the scheduler, not only when
  the scheduler starts. Runs that crash after model load are surfaced with a
  failure type and the last `BENCHMARK_STAGE` marker.

## Chain Ownership

- `paper_idea_agent.py`: produces problem-aware paper ideas.
- `experiment_forge.py`: freezes the experiment evidence contract and the
  problem-awareness contract before execution.
- `result_interpreter.py`: carries problem awareness from experiment outcomes
  into result packets.
- `manuscript_pipeline.py`: builds the canonical manuscript input state.
- `paper_orchestra_prompts.py` and `paperorchestra/full_pipeline.py`: enforce
  the problem-motivation-method-result spine during writing.
- `paperorchestra/figure_orchestra.py`: separates early data figures from
  post-writing API diagrams.
- `paper_orchestra_pipeline.py`: creates ICLR 2026 bundles and records quality
  reports against the local PDF corpus, including `reference_corpus_audit_v1`.
- `reference_corpus_audit.py`: profiles local reference PDFs and generated TeX
  manuscripts so manuscript quality gates compare drafts to the real corpus
  rather than only to generic heuristics.
- `ssh_gpu_backend.py`, `validation_loop.py`, and `gpu_scheduler.py`: keep GPU
  execution aligned with the benchmark contract and recover stale remote runs.
