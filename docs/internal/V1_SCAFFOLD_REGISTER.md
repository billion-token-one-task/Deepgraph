# V1 scaffold register: hand-written priors awaiting a learned replacement

Every hand-set constant in the measurement layer is listed here, with the
evidence that set it and what should eventually replace it. Rule: a new
hand-written constant lands in the same commit that adds it to this table.
Statistical gates (p < 0.05, nonzero baseline, cross-vendor evaluation) are
scientific discipline, not scaffolding, and do not belong here.

| # | Constant | Where | Value | Evidence | V2 replacement |
|---|---|---|---|---|---|
| 1 | GENERATIVE_QA_MIN_SAMPLE_CAP | meta_harness/runner_capability.py | 200 | M0 probe 2026-08-17: n=200 gives 7-point MDE at power 0.8 (r=0.25); run 153's n=4 could distinguish nothing | power analysis computed per-run from the baseline arm |
| 2 | NUMERIC_ANSWER_DATASETS | meta_harness/runner_capability.py | {openai/gsm8k} | string equality against chain-of-thought output scored 0/24 on run 153; numeric extraction verified on 200 hand-checked M0 generations | learned/inspected answer-type detection per dataset |
| 3 | generative max_new_tokens default | meta_harness/runners/generic_transformers.py | 512 | M0 probe: at 256, 59% of GSM8K generations truncated; truncated acc 9.3% vs 61.0% completed | per-task budget search, or generate-until-eos with a cost cap |
| 4 | last-number answer extraction | meta_harness/runner_contract.py `_numeric` | regex last number | M0 probe eyeball of 200 outputs: final number is the answer for GSM8K-style CoT | learned evaluator / structured answer channel |
| 5 | literature effect band 3-10 pts | ops M0 analysis (feasibility branch rule) | 0.03-0.10 | declared prior over published prompt/adapter interventions on small-model GSM8K | meta-analysis over harvested literature |
