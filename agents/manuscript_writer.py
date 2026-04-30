"""Generate grounded manuscript artifacts from completed SciForge runs."""
from __future__ import annotations

import json
import re
from pathlib import Path

from agents.artifact_manager import artifact_path, ensure_artifact_dirs, list_artifacts, record_artifact
from db import database as db


TERMINAL_VERDICTS = {"confirmed", "refuted", "inconclusive"}
TERMINAL_RUN_STATUSES = {"failed"}


def _load_json(value, default):
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _citation_key(paper_id: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_:-]+", "_", str(paper_id)).strip("_")
    return key or "source"


def _strip_sentence_period(text) -> str:
    return str(text or "").strip().rstrip(".")


def _supporting_papers(insight: dict) -> list[dict]:
    raw = _load_json(insight.get("supporting_papers"), [])
    papers = []
    if not isinstance(raw, list):
        return papers
    for idx, item in enumerate(raw, start=1):
        if isinstance(item, dict):
            paper_id = item.get("id") or item.get("paper_id") or f"source_{idx}"
            title = item.get("title") or str(paper_id)
            authors = item.get("authors") or []
        else:
            paper_id = str(item)
            title = str(item)
            authors = []
        papers.append({
            "id": str(paper_id),
            "key": _citation_key(str(paper_id)),
            "title": str(title),
            "authors": authors if isinstance(authors, list) else [str(authors)],
        })
    return papers


def _write_references(papers: list[dict]) -> str:
    entries = []
    for paper in papers:
        author = " and ".join(paper["authors"]) if paper["authors"] else "Unknown"
        entries.append(
            "@misc{" + paper["key"] + ",\n"
            f"  title = {{{paper['title']}}},\n"
            f"  author = {{{author}}},\n"
            f"  note = {{{paper['id']}}}\n"
            "}\n"
        )
    return "\n".join(entries)


def _domain_reference_papers(benchmark_config: dict) -> list[dict]:
    if not _is_safe_rl_cmdp(benchmark_config):
        return []
    return [
        {
            "id": "Altman1999CMDP",
            "key": "Altman1999CMDP",
            "title": "Constrained Markov Decision Processes",
            "authors": ["Eitan Altman"],
        },
        {
            "id": "Puterman1994MDP",
            "key": "Puterman1994MDP",
            "title": "Markov Decision Processes: Discrete Stochastic Dynamic Programming",
            "authors": ["Martin L. Puterman"],
        },
        {
            "id": "Achiam2017CPO",
            "key": "Achiam2017CPO",
            "title": "Constrained Policy Optimization",
            "authors": ["Joshua Achiam", "David Held", "Aviv Tamar", "Pieter Abbeel"],
        },
    ]


def _load_artifact_json(workdir: Path, relative_path: str, default):
    path = artifact_path(workdir, relative_path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return default


def _metric_definition(metric: str, statistical_report: dict | None = None) -> str:
    fairness_penalty = (statistical_report or {}).get("fairness_penalty", 0.45)
    definitions = {
        "fairness_score": (
            "`fairness_score` is the proxy objective emitted by the experiment harness. "
            "For the Fairlearn proxy benchmark it is computed as "
            f"`accuracy - {fairness_penalty} * demographic_parity_gap`, so higher values indicate a better "
            "accuracy/fairness trade-off under this local benchmark."
        ),
        "accuracy": "`accuracy` is the fraction of evaluated examples predicted correctly.",
        "safe_return": (
            "`safe_return` is the finite-CMDP scalar objective emitted by the safe-RL harness: "
            "discounted reward minus `6.0 * max(0, discounted cost - cost_limit)`, so "
            "higher values indicate better reward/cost trade-offs under the configured cost limit."
        ),
        "reward": "`reward` is the expected discounted reward of the evaluated stationary policy.",
        "cost": "`cost` is the expected discounted safety cost of the evaluated stationary policy.",
        "constraint_violation": (
            "`constraint_violation` is `max(0, discounted cost - cost_limit)` for the finite CMDP; "
            "zero means the policy satisfies the recorded safety budget within numerical tolerance."
        ),
    }
    return definitions.get(metric, f"`{metric}` is the primary metric emitted by the recorded experiment harness.")


def _is_safe_rl_cmdp(benchmark_config: dict | None) -> bool:
    return (benchmark_config or {}).get("capability") == "safe_rl_cmdp"


def _method_descriptions(methods: list[str], benchmark_config: dict | None = None) -> str:
    if _is_safe_rl_cmdp(benchmark_config):
        descriptions = {
            "reward_only": "A stationary-policy baseline selected by maximizing expected discounted reward without enforcing the safety-cost limit.",
            "lagrangian_grid_best": "An oracle grid-selected Lagrangian baseline that evaluates the same configured penalty grid and selects the best `safe_return`, included to separate tuning from any preference-cone naming.",
            "preference_cone_policy": "An oracle grid-selected preference-cone penalty selector over reward/cost scalarizations that chooses the policy with the best `safe_return` among configured penalties.",
            "deterministic_feasible_best": "An exhaustive deterministic stationary-policy reference that selects the highest-reward feasible deterministic policy.",
            "occupancy_lp_optimal": "An exact occupancy-measure linear-programming reference for the finite CMDP that can return stationary randomized policies under the cost constraint.",
            "occupancy_enumeration": "A deterministic-policy enumeration audit baseline; it is not treated as the exact constrained CMDP optimum when randomization can help.",
        }
        lines = ["## Implemented Methods", ""]
        for method in methods:
            if method.startswith("lagrangian_penalty_"):
                description = "A fixed-penalty Lagrangian policy selected by maximizing discounted reward minus penalty-weighted discounted safety cost."
            else:
                description = descriptions.get(method, "Configured finite-CMDP benchmark method.")
            lines.append(f"- `{method}`: {description}")
        return "\n".join(lines) + "\n"

    descriptions = {
        "logistic_regression": "A standard logistic regression classifier used as the unconstrained accuracy-oriented baseline.",
        "exponentiated_gradient": "Fairlearn's reductions-based constrained classifier with a demographic-parity constraint.",
        "threshold_optimizer": "Fairlearn's post-processing threshold optimizer for demographic parity.",
        "validation_selected_fairlearn_baseline": "A training-only validation selector over logistic regression, Fairlearn exponentiated-gradient, and Fairlearn threshold-optimizer baselines under the same scalarized validation metric.",
        "preference_cone_threshold": "A group-aware threshold search that optimizes the configured accuracy/fairness objective on training data.",
        "validation_selected_preference_cone": "A training-only validation selector that chooses between logistic regression and preference-cone threshold variants before test evaluation.",
    }
    lines = ["## Implemented Methods", ""]
    for method in methods:
        lines.append(f"- `{method}`: {descriptions.get(method, 'Configured benchmark method.')}")
    return "\n".join(lines) + "\n"


def _algorithmic_specification(benchmark_config: dict) -> str:
    methods = set(benchmark_config.get("methods") or [])
    if not methods:
        return ""
    lines = ["## Algorithmic Specification", ""]
    if _is_safe_rl_cmdp(benchmark_config):
        lines.extend([
            "Each environment is a small discounted Finite CMDP with tabular transitions, rewards, "
            "costs, a start-state distribution, discount factor, and scalar safety-cost limit. "
            "For any stationary policy pi, the harness computes discounted reward and discounted "
            "cost exactly by solving the Bellman linear systems for the induced Markov chain.",
            "",
        ])
        if "preference_cone_policy" in methods:
            penalties = benchmark_config.get("preference_penalties") or [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
            lines.extend([
                "`preference_cone_policy` evaluates a configured penalty grid "
                f"({', '.join(str(item) for item in penalties)}) and selects the policy with the "
                "best `safe_return`, with feasibility determined by `constraint_violation` against "
                "the environment cost limit. This is reported as an oracle grid-selected baseline, "
                "not as a deployable non-oracle tuning protocol.",
                "",
            ])
        if "lagrangian_grid_best" in methods:
            penalties = benchmark_config.get("preference_penalties") or [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
            lines.extend([
                "`lagrangian_grid_best` uses the same configured penalty grid "
                f"({', '.join(str(item) for item in penalties)}) and is likewise an oracle "
                "grid-selected baseline under the reported `safe_return` scalar.",
                "",
            ])
        if "occupancy_lp_optimal" in methods:
            lines.extend([
                "`occupancy_lp_optimal` solves the discounted occupancy-measure linear program "
                "with nonnegative state-action occupancy variables, Bellman flow equalities, and "
                "one discounted cost inequality. The policy recovered from the occupancy table may "
                "be stationary randomized, so this is the exact constrained reference for these "
                "finite CMDPs. In unnormalized discounted occupancy variables x(s,a), the solver "
                "maximizes sum_s,a x(s,a) r(s,a) subject to sum_a x(s,a) - gamma sum_s',a' "
                "P(s | s',a') x(s',a') = rho_0(s), sum_s,a x(s,a)c(s,a) <= cost_limit, and x >= 0. "
                "This LP is the hard-constrained reward optimum; `safe_return` is used for common "
                "reporting across feasible and infeasible methods, not as the LP objective.",
                "",
                "`artifacts/results/lp_validation.json` records SciPy HiGHS backend cross-checks, "
                "flow/cost/objective tolerances, deterministic feasible-policy comparisons, and an "
                "analytic one-state randomized CMDP check whose known optimum mixes two actions 50/50.",
                "",
            ])
        if "occupancy_enumeration" in methods:
            lines.extend([
                "`occupancy_enumeration` enumerates stationary deterministic policies and is kept "
                "as an occupancy-enumeration audit check. It is not used as the exact constrained "
                "frontier when randomized policies are possible. This is not a deep-RL result.",
                "",
            ])
        if any(method.startswith("lagrangian_penalty_") for method in methods):
            lines.extend([
                "Fixed Lagrangian baselines solve the same tabular planning problem after "
                "subtracting a penalty-weighted safety cost from reward.",
                "",
            ])
        return "\n".join(lines).rstrip() + "\n"

    if "preference_cone_threshold" in methods or "validation_selected_preference_cone" in methods:
        fairness_penalty = benchmark_config.get("fairness_penalty", 0.45)
        lines.extend([
            "Preference-cone thresholding fits a probabilistic classifier on the training split, "
            "then searches group-specific decision thresholds on training data only. For each "
            "candidate threshold tuple theta, it computes:",
            "",
            f"`score(theta) = accuracy(theta) - {fairness_penalty} * demographic_parity_gap(theta)`.",
            "",
            "The selected threshold tuple is `argmax_theta score(theta)`, with ties broken toward "
            "lower demographic-parity gap and then higher accuracy. At test time, the frozen "
            "thresholds are applied to held-out examples; test labels are not used for threshold "
            "selection. For large groups, the candidate threshold grid is quantile-capped at "
            "128 threshold candidates per group to avoid quadratic scans over every unique score.",
            "",
        ])
    if "validation_selected_preference_cone" in methods:
        lines.extend([
            "`validation_selected_preference_cone` holds out a training-only validation split, "
            "selects between logistic regression and configured preference-cone penalties on "
            "that validation split, then refits the selected family on the full training split "
            "before test evaluation.",
            "",
        ])
    if "exponentiated_gradient" in methods:
        lines.append("`exponentiated_gradient` is Fairlearn's reductions baseline under a demographic-parity constraint.")
    if "threshold_optimizer" in methods:
        lines.append("`threshold_optimizer` is Fairlearn's post-processing baseline for demographic parity.")
    if "validation_selected_fairlearn_baseline" in methods:
        lines.append("`validation_selected_fairlearn_baseline` selects among Fairlearn baselines on a training-only validation split using the same scalarized objective before final test evaluation.")
    return "\n".join(lines).rstrip() + "\n"


def _train_test_protocol(benchmark_config: dict) -> str:
    datasets = benchmark_config.get("datasets") or []
    seeds = benchmark_config.get("seeds") or []
    dataset_text = ", ".join(f"`{item}`" for item in datasets) if datasets else "the configured datasets"
    seed_text = f"{len(seeds)} configured seeds" if seeds else "the configured seeds"
    if _is_safe_rl_cmdp(benchmark_config):
        return (
            "## Train/Validation/Test Protocol\n\n"
            f"The benchmark evaluates {dataset_text} over {seed_text}. For this finite-CMDP phase, "
            "there is no supervised train/test split: each seed instantiates a deterministic tabular "
            "environment, every configured method plans on the recorded model, and metrics are "
            "computed by exact discounted policy evaluation. The result is a small-model planning "
            "benchmark and not a deep-RL result.\n"
        )
    return (
        "## Train/Validation/Test Protocol\n\n"
        f"The benchmark evaluates {dataset_text} over {seed_text}. Each dataset factory creates "
        "a deterministic train/test split for the seed. Methods may use training labels and "
        "training sensitive attributes for fitting and threshold selection; test labels are not "
        "used for model selection. Metrics are computed once on the held-out test split and "
        "written to `benchmark_results.json` before statistical aggregation.\n"
    )


def _related_work_context(benchmark_config: dict) -> str:
    methods = benchmark_config.get("methods") or []
    baseline_text = ", ".join(f"`{method}`" for method in methods) if methods else "the configured baselines"
    if _is_safe_rl_cmdp(benchmark_config):
        return (
            "## Related Work Context\n\n"
            "This study is scoped to constrained Markov decision processes and safe-RL planning "
            "benchmarks with known finite models, following the classical CMDP occupancy-measure "
            "view [@Altman1999CMDP] and tabular dynamic-programming framing [@Puterman1994MDP]. "
            "The relevant comparison class is therefore "
            f"reward-only planning, fixed Lagrangian scalarizations, and exact occupancy-measure LP "
            f"references such as {baseline_text}; neural constrained-policy methods such as "
            "constrained policy optimization [@Achiam2017CPO] are related but outside this "
            "tabular exact-planning scope. The contribution claimed here is limited to the "
            "recorded finite-CMDP benchmark behavior; it does not claim a new deep-RL algorithm or "
            "a large-scale simulator result.\n"
        )
    return (
        "## Related Work Context\n\n"
        "This study treats group-specific thresholding as a post-processing family for "
        "demographic-parity trade-offs. The relevant comparison class is therefore not only "
        "unconstrained logistic regression, but also reductions-based and post-processing "
        f"fairness baselines such as {baseline_text}. The contribution claimed here is limited "
        "to the recorded benchmark behavior of the implemented threshold search; it is not a "
        "claim that thresholding or demographic-parity optimization is new.\n"
    )


def _introduction_context(benchmark_config: dict, title: str) -> str:
    if _is_safe_rl_cmdp(benchmark_config):
        return (
            "Finite constrained Markov decision processes are useful for checking safe-RL "
            "planning logic because the transition model is explicit, reward/cost trade-offs "
            "can be evaluated exactly, and occupancy-measure LP solutions provide a reference "
            "point for small instances. This note therefore treats the benchmark as an auditable "
            "artifact test: it asks whether the local pipeline can report feasible policies, "
            "unsafe reward-only behavior, tuned Lagrangian baselines, exact LP references, "
            "sensitivity runs, and numerical residuals without claiming a new deep-RL algorithm."
        )
    return f"This manuscript was generated from a DeepGraph deep insight and its experiment artifacts for {title}."


def _statistical_procedure(statistical_report: dict) -> str:
    if not statistical_report:
        return ""
    if statistical_report.get("primary_metric") == "safe_return":
        return (
            "## Statistical Procedure\n\n"
            "For each finite CMDP, method, and seed, the benchmark records one exact policy-evaluation "
            "row. Mean estimates are aggregated across seeds. Confidence intervals in the statistical "
            "report are 95% percentile bootstrap intervals over 1000 seed-level resamples using "
            "the reporter's fixed resampling seed 13. Pairwise comparisons use a two-sided paired sign test over "
            "matched seeds, and secondary tables report reward, cost, and `constraint_violation` "
            "so the safety trade-off is visible rather than hidden inside `safe_return`. The "
            "pairwise tests are descriptive and are not claimed as multiplicity-corrected confirmatory tests.\n"
        )
    return (
        "## Statistical Procedure\n\n"
        "For each dataset, method, and seed, the benchmark records one held-out test metric row. "
        "Mean estimates are aggregated across seeds. Confidence intervals in the statistical "
        "report are percentile bootstrap intervals over seed-level values using the reporter's "
        "fixed resampling seed. Pairwise comparisons use a two-sided paired sign test over "
        "matched seeds; the pairwise table is reported so readers can distinguish strong "
        "logistic-regression gains from weaker comparisons against fairness-aware baselines.\n"
    )


def _dataset_environment_details(benchmark_config: dict, statistical_report: dict,
                                 lp_validation: dict | None = None) -> str:
    datasets = benchmark_config.get("datasets") or []
    seeds = benchmark_config.get("seeds") or []
    methods = benchmark_config.get("methods") or []
    if _is_safe_rl_cmdp(benchmark_config):
        randomized_by_dataset = {}
        if isinstance(lp_validation, dict):
            for item in lp_validation.get("deterministic_reference_comparisons") or []:
                dataset = str(item.get("dataset") or "")
                gap = float(item.get("lp_vs_deterministic_feasible_reward_gap") or item.get("randomization_reward_gap") or 0.0)
                entropy = float(item.get("lp_policy_entropy") or 0.0)
                current = randomized_by_dataset.setdefault(dataset, {"improved": 0, "checked": 0})
                current["checked"] += 1
                if gap > 1e-9 or entropy > 1e-9:
                    current["improved"] += 1
        metadata_rows = []
        try:
            from benchmarks.safe_rl_cmdp.envs import make_cmdp

            for dataset in datasets:
                env = make_cmdp(str(dataset), int(seeds[0]) if seeds else 0)
                observed = randomized_by_dataset.get(env.name)
                if observed and observed["checked"]:
                    randomized_note = (
                        f"LP randomized improvement observed in {observed['improved']}/{observed['checked']} checked seeds"
                    )
                elif env.name == "randomized_bandit":
                    randomized_note = "analytic construction can require randomized stationary policy"
                else:
                    randomized_note = "LP randomization status not summarized in manuscript"
                metadata_rows.append({
                    "dataset": env.name,
                    "states": env.n_states,
                    "actions": env.n_actions,
                    "gamma": env.gamma,
                    "cost_limit": env.cost_limit,
                    "randomization": randomized_note,
                })
        except Exception:
            metadata_rows = []

        lines = [
            "## Dataset And Environment Details",
            "",
            f"Finite CMDP environments: {', '.join(f'`{item}`' for item in datasets) if datasets else 'not recorded'}.",
            "",
            f"Seeds: {', '.join(str(seed) for seed in seeds) if seeds else 'not recorded'}.",
            "",
            f"Methods: {', '.join(f'`{item}`' for item in methods) if methods else 'not recorded'}.",
            "",
            f"Primary metric direction: `{statistical_report.get('metric_direction') or benchmark_config.get('metric_direction') or 'not recorded'}`.",
            "",
            "Each benchmark instance records tabular transition probabilities, reward values, safety costs, a start-state distribution, discount factor, and cost limit in `benchmarks/safe_rl_cmdp/envs.py`; the generated artifact `artifacts/results/cmdp_environment_appendix.json` serializes those quantities for every configured environment/seed. The companion `artifacts/results/lp_validation.json` includes backend cross-checks and an analytic one-state randomized CMDP sanity check for the LP solver. The environments are intentionally small so deterministic enumeration and exact occupancy-measure LP checks remain auditable.",
            "",
        "Environment construction summary: `randomized_bandit` is a one-state CMDP where the constrained optimum can require randomization; `risky_shortcut` contrasts a high-reward costly shortcut with a safer branch; `delayed_safety` delays cost consequences across a four-state chain; `resource_gathering` uses stochastic resource transitions; `stochastic_bridge` adds stochastic bridge outcomes over five states; `tight_budget_chain` uses a five-state chain with a tight cumulative-cost budget. Seeds perturb rewards and costs with small deterministic Gaussian jitter while preserving the recorded transition topology.",
            "",
            "Systematic generated environments named `systematic_*` vary state count, action count, discount factor, cost limit, transition stochasticity, and randomized-optimum regimes using a deterministic specification in `benchmarks/safe_rl_cmdp/envs.py`. They are included to test whether the harness and reports remain stable beyond the original hand-designed examples; they are still small tabular CMDPs rather than external simulator benchmarks.",
            "",
            "No neural policy, simulator roll-out estimator, or learned dynamics model is used in this phase; the reported result is not a deep-RL result. The primary scalar `safe_return` uses the harness penalty on positive constraint violation and is reported alongside raw discounted reward, discounted cost, and `constraint_violation`.",
        ]
        if metadata_rows:
            lines.extend([
                "",
                "### Environment Metadata Summary",
                "",
                "| Environment | States | Actions | Gamma | Cost limit | Randomized optimum note |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ])
            for row in metadata_rows:
                lines.append(
                    f"| {row['dataset']} | {row['states']} | {row['actions']} | "
                    f"{float(row['gamma']):.3f} | {float(row['cost_limit']):.6f} | "
                    f"{row['randomization']} |"
                )
        return "\n".join(lines) + "\n"

    lines = [
        "## Dataset And Environment Details",
        "",
        f"Datasets: {', '.join(f'`{item}`' for item in datasets) if datasets else 'not recorded'}.",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds) if seeds else 'not recorded'}.",
        "",
        f"Methods: {', '.join(f'`{item}`' for item in methods) if methods else 'not recorded'}.",
        "",
        f"Primary metric direction: `{statistical_report.get('metric_direction') or benchmark_config.get('metric_direction') or 'not recorded'}`.",
        "",
        "Preprocessing is performed by the dataset factory: tabular categorical columns are one-hot encoded, numeric columns are standardized, the protected attribute is appended as the final feature for methods that require group-specific thresholds, and the target/protected columns are excluded from the ordinary feature matrix.",
        "",
        "For ordinary prediction, the implementation removes the appended protected-attribute column before every logistic-regression and Fairlearn estimator fit. The protected attribute is passed separately as `sensitive_features` for Fairlearn constraints and is used by preference-cone methods only for group-specific threshold routing.",
    ]
    return "\n".join(lines) + "\n"


def _tradeoff_interpretation(statistical_report: dict) -> str:
    if not statistical_report:
        return ""
    best_method = statistical_report.get("best_method") or "candidate"
    baseline_method = statistical_report.get("baseline_method") or "baseline"
    summaries = statistical_report.get("metric_summaries") or []

    if statistical_report.get("primary_metric") == "safe_return":
        def means_for_safe(metric: str, method: str) -> list[float]:
            return [
                float(item["mean"]) for item in summaries
                if item.get("metric") == metric and item.get("method") == method and item.get("mean") is not None
            ]

        notes = []
        for metric, direction, label in [
            ("reward", "higher", "discounted reward"),
            ("cost", "lower", "discounted safety cost"),
            ("constraint_violation", "lower", "constraint violation"),
        ]:
            cand = means_for_safe(metric, best_method)
            base = means_for_safe(metric, baseline_method)
            if not cand or not base:
                continue
            cand_mean = sum(cand) / len(cand)
            base_mean = sum(base) / len(base)
            delta = cand_mean - base_mean
            if direction == "lower":
                delta = -delta
            direction_text = "improves" if delta >= 0 else "worsens"
            notes.append(
                f"- Against `{baseline_method}`, `{best_method}` {direction_text} average {label} "
                f"under the reported aggregation (signed improvement {delta:+.6f})."
            )
        if not notes:
            notes.append("- Reward/cost secondary metric trade-offs were not available in the statistical report.")
        return (
            "## Trade-off Interpretation\n\n"
            "`safe_return` is a scoped benchmark scalarization, not a universal safe-RL ranking. "
            "It is reported together with discounted reward, discounted cost, and "
            "`constraint_violation` to show whether gains come from higher reward, lower safety "
            "cost, or both. The exact occupancy-measure LP method is an upper reference point for "
            "these finite models, not the proposed method.\n\n"
            + "\n".join(notes)
            + "\n"
        )

    def means_for(metric: str, method: str) -> list[float]:
        return [
            float(item["mean"]) for item in summaries
            if item.get("metric") == metric and item.get("method") == method and item.get("mean") is not None
        ]

    notes = []
    for metric, direction, label in [
        ("accuracy", "higher", "accuracy"),
        ("demographic_parity_gap", "lower", "demographic-parity gap"),
        ("equalized_odds_gap", "lower", "equalized-odds gap"),
    ]:
        cand = means_for(metric, best_method)
        base = means_for(metric, baseline_method)
        if not cand or not base:
            continue
        cand_mean = sum(cand) / len(cand)
        base_mean = sum(base) / len(base)
        delta = cand_mean - base_mean
        if direction == "lower":
            delta = -delta
        direction_text = "improves" if delta >= 0 else "worsens"
        notes.append(
            f"- Against `{baseline_method}`, `{best_method}` {direction_text} average {label} "
            f"under the reported aggregation (signed improvement {delta:+.6f})."
        )
    if not notes:
        notes.append("- Secondary metric trade-offs were not available in the statistical report.")
    return (
        "## Trade-off Interpretation\n\n"
        "The scalarized primary metric is not interpreted as a universal fairness ranking. "
        "It is reported together with accuracy, demographic-parity gap, and equalized-odds gap "
        "to expose where a demographic-parity gain comes with other costs.\n\n"
        + "\n".join(notes)
        + "\n"
    )


def _iteration_table(iterations: list[dict]) -> str:
    if not iterations:
        return (
            "Benchmark-suite mode does not run the iterative code-edit loop. Audit evidence is "
            "recorded in `artifacts/results/benchmark_results.json`, "
            "`artifacts/results/statistical_report.json`, and "
            "`artifacts/results/evidence_gate.json`.\n"
        )
    lines = ["| Iteration | Phase | Status | Metric |", "| --- | --- | --- | --- |"]
    for item in iterations[:20]:
        lines.append(
            f"| {item.get('iteration_number', '')} | "
            f"{item.get('phase', '')} | "
            f"{item.get('status', '')} | "
            f"{item.get('metric_value', '')} |"
        )
    return "\n".join(lines) + "\n"


def _claim_lines(claims: list[dict]) -> str:
    if not claims:
        return "- No experimental claim rows were available.\n"
    lines = []
    for claim in claims:
        verdict = claim.get("verdict", "inconclusive")
        lines.append(f"- [{verdict}] {claim.get('claim_text', '')}")
    return "\n".join(lines) + "\n"


def _audit_artifacts(benchmark_config: dict) -> str:
    if benchmark_config:
        artifacts = [
            "- `benchmark_config.json`",
            "- `research_spec.json`",
            "- `artifacts/results/benchmark_results.json`",
            "- `artifacts/results/statistical_report.json`",
            "- `artifacts/results/evidence_gate.json`",
            "- `artifacts/tables/main_results.md`",
        ]
        if _is_safe_rl_cmdp(benchmark_config):
            artifacts.extend([
                "- `artifacts/results/cmdp_environment_appendix.json`",
                "- `artifacts/results/lp_validation.json`",
                "- `artifacts/results/reproduction_manifest.json`",
            ])
        return "\n".join(artifacts)
    return "\n".join([
        "- `artifacts/results/metrics.json`",
        "- `artifacts/results/iterations.json`",
        "- `artifacts/logs/run.log`",
    ])


def _aggregate_method_summary(statistical_report: dict) -> str:
    primary_metric = statistical_report.get("primary_metric") or "metric"
    aggregate_rows = [
        item for item in (statistical_report.get("aggregate_metric_summaries") or [])
        if item.get("metric") == primary_metric
    ]
    source_note = "across all configured datasets and seeds"
    if not aggregate_rows:
        aggregate_rows = statistical_report.get("summary") or []
        source_note = "using the available per-dataset summary rows"
    if not aggregate_rows:
        return ""

    lines = [
        "### Aggregate Method Summary",
        "",
        f"The table reports `{primary_metric}` {source_note}. It includes reference methods such as exact LP solvers when they are configured, so the deployed/candidate comparison is not confused with the absolute reference frontier.",
        "",
        "| Method | Metric | Mean | CI | n |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in sorted(aggregate_rows, key=lambda row: str(row.get("method", "")))[:80]:
        lines.append(
            f"| {item.get('method', '')} | {item.get('metric', primary_metric)} | "
            f"{item.get('mean', '')} | [{item.get('ci_low', '')}, {item.get('ci_high', '')}] | "
            f"{item.get('n', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _aggregate_safety_metric_summary(statistical_report: dict) -> str:
    aggregate_rows = statistical_report.get("aggregate_metric_summaries") or []
    metrics = {"reward", "cost", "constraint_violation"}
    rows = [item for item in aggregate_rows if item.get("metric") in metrics]
    if not rows:
        return ""
    lines = [
        "### Aggregate Reward Cost And Violation Summary",
        "",
        "This compact table reports the aggregate secondary metrics used to interpret `safe_return`. Lower `cost` and `constraint_violation` are better; higher `reward` is better.",
        "",
        "| Method | Metric | Mean | CI | n |",
        "| --- | --- | --- | --- | --- |",
    ]
    order = {"reward": 0, "cost": 1, "constraint_violation": 2}
    for item in sorted(rows, key=lambda row: (str(row.get("method", "")), order.get(str(row.get("metric")), 99)))[:120]:
        lines.append(
            f"| {item.get('method', '')} | {item.get('metric', '')} | "
            f"{item.get('mean', '')} | [{item.get('ci_low', '')}, {item.get('ci_high', '')}] | "
            f"{item.get('n', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _runtime_summary(statistical_report: dict) -> str:
    rows = [
        item for item in (statistical_report.get("aggregate_metric_summaries") or [])
        if item.get("metric") == "runtime_seconds"
    ]
    if not rows:
        return ""
    lines = [
        "### Runtime Summary",
        "",
        "Runtime is measured per environment/seed/method row by the benchmark harness. These small finite CMDPs are intended for auditability, not scalability claims.",
        "",
        "| Method | Mean seconds | CI | n |",
        "| --- | ---: | --- | ---: |",
    ]
    for item in sorted(rows, key=lambda row: str(row.get("method", "")))[:40]:
        lines.append(
            f"| {item.get('method', '')} | {item.get('mean', '')} | "
            f"[{item.get('ci_low', '')}, {item.get('ci_high', '')}] | {item.get('n', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _safety_penalty_sensitivity_summary(statistical_report: dict) -> str:
    rows = [
        item for item in (statistical_report.get("ablation_summary") or [])
        if "safety_penalty=" in str(item.get("ablation") or "") and item.get("mean") is not None
    ]
    if not rows:
        return ""
    groups = {}
    for item in rows:
        label = str(item.get("ablation"))
        groups.setdefault(label, []).append(float(item.get("mean")))
    lines = [
        "### Safety Penalty Sensitivity Summary",
        "",
        "Safety-penalty ablations are summarized separately by configured penalty value instead of being pooled into one row group.",
        "",
        "| Ablation | Mean of environment means | Min environment mean | Max environment mean | Environments |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, values in sorted(groups.items()):
        lines.append(
            f"| {label} | {sum(values) / len(values)} | {min(values)} | {max(values)} | {len(values)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _environment_group_summary(statistical_report: dict) -> str:
    primary_metric = statistical_report.get("primary_metric") or "metric"
    summary = [
        item for item in (statistical_report.get("summary") or [])
        if item.get("metric") == primary_metric and item.get("mean") is not None
    ]
    if not summary:
        return ""
    grouped = {}
    for item in summary:
        grouped.setdefault(str(item.get("method")), []).append(float(item.get("mean")))
    lines = [
        "### Environment-Grouped Summary",
        "",
        "This table treats environments as groups by averaging per-environment means. It is an audit summary for heterogeneous CMDPs, not a claim that all dataset-seed rows are exchangeable.",
        "",
        "| Method | Mean of environment means | Min environment mean | Max environment mean | Environments |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method, values in sorted(grouped.items()):
        lines.append(
            f"| {method} | {sum(values) / len(values)} | {min(values)} | {max(values)} | {len(values)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _lp_randomization_gap_summary(lp_validation: dict) -> str:
    if not isinstance(lp_validation, dict):
        return ""
    rows = lp_validation.get("deterministic_reference_comparisons") or []
    if not rows:
        return ""
    grouped = {}
    for item in rows:
        dataset = str(item.get("dataset") or "")
        gap = float(item.get("lp_vs_deterministic_feasible_reward_gap") or item.get("randomization_reward_gap") or 0.0)
        bucket = grouped.setdefault(dataset, {"checked": 0, "improved": 0, "max_gap": 0.0})
        bucket["checked"] += 1
        bucket["max_gap"] = max(bucket["max_gap"], gap)
        if gap > 1e-9:
            bucket["improved"] += 1
    lines = [
        "### LP Randomization Gap Summary",
        "",
        "| Environment | Seeds checked | LP improves over deterministic feasible | Max reward gap |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dataset, item in sorted(grouped.items()):
        lines.append(
            f"| {dataset} | {item['checked']} | {item['improved']} | {item['max_gap']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _lp_validation_summary(lp_validation: dict) -> str:
    if not isinstance(lp_validation, dict) or not lp_validation:
        return ""
    rows = lp_validation.get("deterministic_reference_comparisons") or []
    cross_checks = lp_validation.get("solver_backend_cross_checks") or []

    def max_abs(name: str):
        values = []
        for row in rows:
            value = row.get(name)
            if value is not None:
                values.append(abs(float(value)))
        return max(values) if values else None

    backend_gaps = [
        abs(float(item.get("max_objective_gap")))
        for item in cross_checks
        if item.get("max_objective_gap") is not None
    ]
    analytic = lp_validation.get("analytic_randomized_checks") or []
    lines = [
        "## LP Validation Summary",
        "",
        f"LP validation status: `{lp_validation.get('status', 'unknown')}`.",
        "",
        f"Instances checked: `{len(rows)}`.",
        "",
        f"Maximum flow residual: `{max_abs('lp_flow_residual')}`.",
        "",
        f"Maximum cost residual: `{max_abs('lp_cost_residual')}`.",
        "",
        f"Maximum objective gap: `{max_abs('lp_objective_gap')}`.",
        "",
        f"Maximum backend objective gap: `{max(backend_gaps) if backend_gaps else None}`.",
        "",
        f"Analytic randomized checks: `{len(analytic)}`.",
        "",
        "The full per-instance LP/deterministic/candidate gap table is stored in `artifacts/results/lp_validation.json`.",
    ]
    gap_summary = _lp_randomization_gap_summary(lp_validation)
    if gap_summary:
        lines.extend(["", gap_summary.strip()])
    return "\n".join(lines) + "\n"


def _reproduction_check_summary(reproduction_check: dict) -> str:
    if not isinstance(reproduction_check, dict) or not reproduction_check:
        return ""
    lines = [
        "## Reproduction Check Summary",
        "",
        f"Local reproduction-check status: `{reproduction_check.get('status', 'unknown')}`.",
        "",
        f"Scope: {reproduction_check.get('scope') or 'not recorded'}.",
        "",
        "| Command | Exit code | Seconds |",
        "| --- | ---: | ---: |",
    ]
    for check in (reproduction_check.get("checks") or [])[:5]:
        command = " ".join(str(part) for part in check.get("command") or [])
        lines.append(
            f"| `{command}` | {check.get('exit_code')} | {check.get('duration_seconds')} |"
        )
    lines.extend([
        "",
        "The full stdout/stderr tails are stored in `artifacts/results/reproduction_check.json`.",
    ])
    return "\n".join(lines) + "\n"


def _reference_candidate_framing(statistical_report: dict, benchmark_config: dict, lp_validation: dict | None = None) -> str:
    if not statistical_report:
        return ""
    candidate = (
        statistical_report.get("candidate_method")
        or benchmark_config.get("candidate_method")
        or statistical_report.get("best_method")
    )
    baseline = statistical_report.get("baseline_method") or benchmark_config.get("baseline_method")
    absolute = statistical_report.get("absolute_best_method")
    reference = benchmark_config.get("reference_method") or absolute
    if not any([candidate, baseline, absolute, reference]):
        return ""

    lines = ["## Reference And Candidate Framing", ""]
    if baseline:
        lines.append(f"- Baseline comparison method: `{baseline}`.")
    if candidate:
        lines.append(f"- Configured candidate method: `{candidate}`.")
    if reference:
        lines.append(f"- Reference method: `{reference}`.")
    if absolute:
        lines.append(f"- Highest aggregate method in the reported table: `{absolute}`.")

    if _is_safe_rl_cmdp(benchmark_config):
        lines.extend([
            "",
            "For this finite-CMDP phase, `occupancy_lp_optimal` is the exact LP reference, not the deployable candidate. It is included to audit the constrained frontier and randomized-policy gap on small tabular CMDPs. Fixed-penalty methods and oracle grid-selected baselines are reported separately, and any non-oracle deployment claim must use the configured candidate method recorded in `benchmark_config.json`.",
            "",
            "Configuration provenance: the candidate method, reference method, safety penalty, method list, dataset list, and seeds are written to `benchmark_config.json` before benchmark execution and repeated in `artifacts/results/reproduction_manifest.json`. The manuscript treats penalty-grid winners as diagnostic/oracle-selected baselines unless a separate train/validation selection protocol is implemented.",
        ])
        if lp_validation:
            lines.extend(["", _lp_validation_summary(lp_validation).strip()])
    elif absolute and candidate and absolute != candidate:
        lines.append("")
        lines.append("The absolute best aggregate method is reported as a reference point; the candidate comparison remains the configured method recorded in `benchmark_config.json`.")
    lines.append("")
    return "\n".join(lines)


def _statistical_evidence(statistical_report: dict) -> str:
    if not isinstance(statistical_report, dict) or not statistical_report:
        return "No benchmark-suite statistical report was available.\n"

    lines = ["## Statistical Evidence", ""]
    primary_metric = statistical_report.get("primary_metric") or "metric"
    baseline_method = statistical_report.get("baseline_method") or "baseline"
    best_method = statistical_report.get("best_method") or "candidate"
    candidate_method = statistical_report.get("candidate_method") or best_method
    lines.append(f"Primary metric: `{primary_metric}`.")
    lines.append("")
    lines.append(f"Baseline method: `{baseline_method}`.")
    lines.append("")
    lines.append(f"Configured candidate method: `{candidate_method}`.")
    lines.append("")
    absolute_best = statistical_report.get("absolute_best_method")
    if absolute_best:
        lines.append(f"Exact/reference best method: `{absolute_best}`.")
        lines.append("")

    aggregate = _aggregate_method_summary(statistical_report)
    if aggregate:
        lines.append(aggregate)
    grouped = _environment_group_summary(statistical_report)
    if grouped:
        lines.append(grouped)
    aggregate_secondary = _aggregate_safety_metric_summary(statistical_report)
    if aggregate_secondary:
        lines.append(aggregate_secondary)
    runtime_summary = _runtime_summary(statistical_report)
    if runtime_summary:
        lines.append(runtime_summary)
    sensitivity_summary = _safety_penalty_sensitivity_summary(statistical_report)
    if sensitivity_summary:
        lines.append(sensitivity_summary)

    comparisons = statistical_report.get("comparisons") or []
    if comparisons:
        lines.extend(["| Dataset | Candidate | Mean Delta | Paired Sign Test p | Wins/Losses/Ties |",
                      "| --- | --- | --- | --- | --- |"])
        for item in comparisons[:20]:
            lines.append(
                f"| {item.get('dataset', '')} | {item.get('candidate', '')} | "
                f"{item.get('mean_delta', '')} | {item.get('paired_sign_test_p', '')} | "
                f"{item.get('wins', '')}/{item.get('losses', '')}/{item.get('ties', '')} |"
            )
        lines.append("")

    pairwise = statistical_report.get("pairwise_comparisons") or []
    if pairwise:
        lines.extend([
            "### Pairwise Baseline Comparisons",
            "",
            "| Dataset | Baseline | Candidate | Metric | Mean Delta | Paired Sign Test p | Wins/Losses/Ties |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ])
        for item in pairwise[:40]:
            lines.append(
                f"| {item.get('dataset', '')} | {item.get('baseline', '')} | "
                f"{item.get('candidate', '')} | {item.get('metric', '')} | "
                f"{item.get('mean_delta', '')} | {item.get('paired_sign_test_p', '')} | "
                f"{item.get('wins', '')}/{item.get('losses', '')}/{item.get('ties', '')} |"
            )
        lines.append("")

    summary = statistical_report.get("summary") or []
    if summary:
        lines.extend(["### Per-Dataset Primary Metric Summary", "",
                      "| Dataset | Method | Mean | CI | n |",
                      "| --- | --- | --- | --- | --- |"])
        for item in summary[:80]:
            lines.append(
                f"| {item.get('dataset', '')} | {item.get('method', '')} | "
                f"{item.get('mean', '')} | [{item.get('ci_low', '')}, {item.get('ci_high', '')}] | "
                f"{item.get('n', '')} |"
            )
        lines.append("")

    metric_summaries = statistical_report.get("metric_summaries") or []
    secondary = [
        item for item in metric_summaries
        if item.get("metric") != primary_metric
    ]
    if secondary:
        lines.extend([
            "## Secondary Metrics",
            "",
            "The manuscript shows a compact secondary-metric excerpt; the complete per-dataset and per-seed rows are in `artifacts/results/statistical_report.json` and `artifacts/results/benchmark_results.json`.",
            "",
            "| Dataset | Method | Metric | Mean | CI | n |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        shown = secondary[:48]
        for item in shown:
            lines.append(
                f"| {item.get('dataset', '')} | {item.get('method', '')} | "
                f"{item.get('metric', '')} | {item.get('mean', '')} | "
                f"[{item.get('ci_low', '')}, {item.get('ci_high', '')}] | "
                f"{item.get('n', '')} |"
            )
        if len(secondary) > len(shown):
            lines.append(f"| ... | ... | ... | {len(secondary) - len(shown)} additional secondary summary rows omitted from manuscript; see statistical_report.json. | ... | ... |")
        lines.append("")

    ablation_summary = statistical_report.get("ablation_summary") or []
    if ablation_summary:
        lines.extend([
            "## Ablation And Sensitivity",
            "",
            "| Ablation | Dataset | Method | Mean | CI | n |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for item in ablation_summary[:30]:
            lines.append(
                f"| {item.get('ablation', '')} | {item.get('dataset', '')} | {item.get('method', '')} | "
                f"{item.get('mean', '')} | [{item.get('ci_low', '')}, {item.get('ci_high', '')}] | "
                f"{item.get('n', '')} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _evidence_gate_section(evidence_gate: dict) -> str:
    if not isinstance(evidence_gate, dict) or not evidence_gate:
        return ""
    blocking = evidence_gate.get("blocking_reasons") or []
    satisfied = evidence_gate.get("satisfied_requirements") or []
    return (
        "## Evidence Gate\n\n"
        "This section is an internal artifact-audit summary, not an external peer-review verdict.\n\n"
        f"Manuscript status: `{evidence_gate.get('manuscript_status') or 'unknown'}`.\n\n"
        "Satisfied requirements:\n"
        + ("\n".join(f"- {item}" for item in satisfied) if satisfied else "- None recorded.")
        + "\n\nBlocking reasons:\n"
        + ("\n".join(f"- {item}" for item in blocking) if blocking else "- None recorded.")
        + "\n"
    )


def _artifact_hash_summary(artifacts: list[dict]) -> str:
    wanted = {
        "benchmark_config.json",
        "research_spec.json",
        "artifacts/results/benchmark_results.json",
        "artifacts/results/statistical_report.json",
        "artifacts/results/lp_validation.json",
        "artifacts/results/cmdp_environment_appendix.json",
        "artifacts/results/reproduction_manifest.json",
        "artifacts/tables/main_results.md",
    }
    rows = [
        item for item in artifacts
        if item.get("path") in wanted and item.get("sha256")
    ]
    if not rows:
        return ""
    lines = [
        "## Artifact Hash Summary",
        "",
        "| Artifact | sha256 |",
        "| --- | --- |",
    ]
    for item in sorted(rows, key=lambda row: row.get("path", "")):
        lines.append(f"| `{item.get('path')}` | `{item.get('sha256')}` |")
    return "\n".join(lines) + "\n"


def _paper_markdown(run: dict, insight: dict, claims: list[dict], papers: list[dict],
                    metrics: dict, iterations: list[dict],
                    statistical_report: dict | None = None,
                    evidence_gate: dict | None = None,
                    benchmark_config: dict | None = None,
                    artifacts: list[dict] | None = None,
                    lp_validation: dict | None = None,
                    reproduction_check: dict | None = None) -> str:
    cite = f" [@{papers[0]['key']}]" if papers else ""
    benchmark_config = benchmark_config or {}
    title = (
        benchmark_config.get("paper_title")
        if (evidence_gate or {}).get("manuscript_status") == "paper_ready_candidate"
        else None
    ) or insight.get("title") or f"Experiment Run {run['id']}"
    scoped_claim = benchmark_config.get("scoped_claim")
    metric = (statistical_report or {}).get("primary_metric") or metrics.get("metric_name") or run.get("baseline_metric_name") or "metric"
    baseline = metrics.get("baseline", run.get("baseline_metric_value"))
    best = metrics.get("best_value", run.get("best_metric_value"))
    candidate_method = (statistical_report or {}).get("candidate_method") or (statistical_report or {}).get("best_method")
    absolute_best_method = (statistical_report or {}).get("absolute_best_method")
    effect = metrics.get("effect_size", run.get("effect_size"))
    effect_pct = metrics.get("effect_pct", run.get("effect_pct"))
    kept = metrics.get("iterations_kept", run.get("iterations_kept"))
    verdict = run.get("hypothesis_verdict") or "inconclusive"
    manuscript_status = (evidence_gate or {}).get("manuscript_status")
    is_candidate = manuscript_status == "paper_ready_candidate"
    claim_label = "Evidence-Gated Benchmark Result" if is_candidate and verdict == "confirmed" else (
        "Proxy-Supported Result" if verdict == "confirmed" else verdict.title()
    )
    abstract_note = (
        "The accompanying artifact package contains the benchmark rows, statistical report, "
        "configuration, and reproducibility metadata used for this scoped artifact-reporting claim."
        if is_candidate
        else (
            "This is a machine-generated research artifact, not a submission-ready paper."
        )
    )
    method_note = (
        "The benchmark suite evaluates the scoped empirical claim using configured offline "
        "datasets, paired baselines, multi-seed statistics, and recorded ablations. A "
        "local evidence-gate status is recorded as an engineering audit signal; "
        "the scientific result remains scoped to the benchmark design and reported trade-offs."
        if is_candidate
        else (
            "The proposed method and experiment were generated from the stored deep insight and "
            "executed through SciForge. This draft reports only metrics present in the experiment "
            "database and artifact files. A `confirmed` verdict here means the proxy validation "
            "loop kept an improving code change under the configured metric; it does not establish "
            "a publishable scientific claim without broader experiments."
        )
    )
    limitations = (
        "The artifact package should be checked against the recorded code, logs, metrics, benchmark "
        "configuration, statistical report, and citations before submission. The evidence "
        "is limited to the implemented offline benchmark capability and should not be generalized "
        "beyond that scope without additional experiments."
        if is_candidate
        else (
            "This draft is machine-generated and requires human review before submission. It must "
            "not be submitted unless the associated code, logs, metrics, statistical power, "
            "baseline definitions, and citation grounding are independently verified. A proxy "
            "benchmark is useful for debugging the discovery engine, but it is not enough by itself "
            "for a publishable research paper."
        )
    )
    return f"""# {title}

## Abstract

We evaluate the scoped claim: {_strip_sentence_period(scoped_claim or insight.get('hypothesis') or insight.get('problem_statement') or title)}. In recorded SciForge benchmark artifacts, the configured candidate has {metric}={best} versus baseline {baseline}, with signed aggregate difference {effect}. This reward-only comparison is a diagnostic sanity check, not evidence that the fixed-penalty candidate is a strong deployable safe-RL method. {abstract_note} Supporting literature is grounded in stored DeepGraph evidence{cite}.

## Introduction

{insight.get('problem_statement') or _introduction_context(benchmark_config, title)}

{_related_work_context(benchmark_config) if benchmark_config else ""}

## Method

{method_note}

{_method_descriptions(benchmark_config.get("methods") or [], benchmark_config) if benchmark_config.get("methods") else ""}
{_algorithmic_specification(benchmark_config)}
{_train_test_protocol(benchmark_config) if benchmark_config else ""}
{_dataset_environment_details(benchmark_config, statistical_report or {}, lp_validation or {}) if benchmark_config else ""}

## Metric Definition

{_metric_definition(str(metric), statistical_report)}

## Experiments

Primary metric: `{metric}`.

Baseline value: `{baseline}`.

Configured candidate value{f" (`{candidate_method}`)" if candidate_method else ""}: `{best}`.

{f"Reference frontier method (not candidate): `{absolute_best_method}`." if absolute_best_method and absolute_best_method != candidate_method else ""}

Effect percent: `{effect_pct}`. This percentage is descriptive and can be hard to interpret when the baseline value is negative; the signed aggregate difference above is the primary descriptive difference and is not a standardized effect size.

Kept hypothesis-testing iterations: `{kept}`.

Audit artifacts:

{_audit_artifacts(benchmark_config)}

{_statistical_evidence(statistical_report or {})}
{_statistical_procedure(statistical_report or {})}
{_tradeoff_interpretation(statistical_report or {})}
{_reference_candidate_framing(statistical_report or {}, benchmark_config, lp_validation or {})}
{_reproduction_check_summary(reproduction_check or {})}
{_artifact_hash_summary(artifacts or [])}
{_evidence_gate_section(evidence_gate or {})}
## Results

### {claim_label}

{_claim_lines(claims)}

## Per-Iteration Audit Trail

{_iteration_table(iterations)}

## Limitations

{limitations}

## References

{chr(10).join(f'- [@{paper["key"]}] {paper["title"]}' for paper in papers) if papers else '- No supporting papers were attached.'}
"""


def _additional_experiments_report(run: dict, insight: dict, evidence_gate: dict) -> str:
    title = insight.get("title") or f"Experiment Run {run['id']}"
    blocking = evidence_gate.get("blocking_reasons") or []
    satisfied = evidence_gate.get("satisfied_requirements") or []
    next_required = evidence_gate.get("next_required_experiments") or []
    return f"""# Additional Experiments Required: {title}

## Evidence Gate Status

Manuscript status: `{evidence_gate.get('manuscript_status') or 'unknown'}`.

## Blocking Reasons

{chr(10).join(f'- {item}' for item in blocking) if blocking else '- None recorded.'}

## Satisfied Requirements

{chr(10).join(f'- {item}' for item in satisfied) if satisfied else '- None recorded.'}

## Next Required Experiments

{chr(10).join(f'- {item}' for item in next_required) if next_required else '- No follow-up experiment was recorded.'}

## Interpretation

The experiment pipeline ran far enough to produce an evidence decision, but the current artifact package is not a publishable paper. Treat this as an execution report and queue the listed follow-up experiments before regenerating a manuscript candidate.
"""


def _preliminary_report(run: dict, insight: dict, evidence_gate: dict) -> str:
    title = insight.get("title") or f"Experiment Run {run['id']}"
    return f"""# Preliminary Report: {title}

## Summary

This run has not produced enough artifact evidence for a paper candidate.

{_evidence_gate_section(evidence_gate)}
"""


def _negative_report(run: dict, insight: dict, claims: list[dict], papers: list[dict]) -> str:
    cite = f" [@{papers[0]['key']}]" if papers else ""
    title = insight.get("title") or f"Experiment Run {run['id']}"
    failure = run.get("error_message") or "No failure message was recorded."
    return f"""# Negative Result: {title}

## Summary

The automated validation run did not confirm the hypothesis: {insight.get('hypothesis') or insight.get('problem_statement') or title}{cite}.

## Experimental Outcome

Baseline value: `{run.get('baseline_metric_value')}`.

Best value: `{run.get('best_metric_value')}`.

Effect size: `{run.get('effect_size')}`.

Failure reason: `{failure}`.

## Claims

{_claim_lines(claims)}

## Interpretation

This report is a negative or refuting result. It should be used to refine the hypothesis or document why the proposed direction did not work under the recorded experimental conditions.
"""


def _reproducibility(run: dict) -> str:
    return f"""# Reproducibility Statement

- Experiment run: `{run['id']}`
- Workdir: `{run.get('workdir')}`
- Status: `{run.get('status')}`
- Verdict: `{run.get('hypothesis_verdict') or 'inconclusive'}`
- Baseline metric: `{run.get('baseline_metric_name')}`
- Baseline value: `{run.get('baseline_metric_value')}`
- Best value: `{run.get('best_metric_value')}`
- Error message: `{run.get('error_message') or ''}`

For benchmark-suite runs, regenerate the core artifacts with:

```powershell
.\\.venv\\Scripts\\python.exe -m unittest tests.test_safe_rl_cmdp_benchmark tests.test_benchmark_suite tests.test_statistical_reporter
```

The run artifacts needed to reproduce the manuscript tables are `benchmark_config.json`, `research_spec.json`, `artifacts/results/benchmark_results.json`, `artifacts/results/statistical_report.json`, `artifacts/tables/main_results.md`, `artifacts/results/cmdp_environment_appendix.json`, `artifacts/results/lp_validation.json`, and `artifacts/results/reproduction_manifest.json` when present. The LP backend is SciPy `linprog(method="highs")`; package versions are recorded in the reproduction manifest and determined by the active virtual environment.
"""


def _is_review_revision_candidate(evidence_gate: dict) -> bool:
    if not isinstance(evidence_gate, dict):
        return False
    if evidence_gate.get("manuscript_status") != "needs_more_experiments":
        return False
    blocking = set(evidence_gate.get("blocking_reasons") or [])
    if blocking != {"review_requires_revision"}:
        return False
    satisfied = set(evidence_gate.get("satisfied_requirements") or [])
    required = {
        "has_benchmark_results",
        "has_baseline_comparison",
        "has_statistical_report",
        "has_multi_seed",
        "has_multi_dataset",
        "has_ablation",
    }
    return required.issubset(satisfied)


def _write_artifact(workdir: Path, run_id: int, relative_path: str, artifact_type: str, text: str) -> str:
    path = artifact_path(workdir, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    record_artifact(workdir, run_id, artifact_type, path)
    return str(path)


def generate_manuscript(run_id: int) -> dict:
    """Generate a grounded manuscript package for a completed experiment run."""
    run = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not run:
        return {"status": "error", "reason": "run_not_found"}

    run_status = run.get("status")
    verdict = run.get("hypothesis_verdict")
    if verdict not in TERMINAL_VERDICTS and run_status not in TERMINAL_RUN_STATUSES:
        return {"status": "error", "reason": "run_not_terminal", "run_id": run_id}
    if verdict not in TERMINAL_VERDICTS:
        verdict = "inconclusive"

    workdir = Path(run.get("workdir") or "")
    if not workdir:
        return {"status": "error", "reason": "missing_workdir", "run_id": run_id}
    ensure_artifact_dirs(workdir)

    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (run["deep_insight_id"],)) or {}
    claims = db.fetchall("SELECT * FROM experimental_claims WHERE run_id=?", (run_id,))
    papers = _supporting_papers(insight)
    metrics = _load_artifact_json(workdir, "artifacts/results/metrics.json", {})
    if not isinstance(metrics, dict):
        metrics = {}
    iterations = _load_artifact_json(workdir, "artifacts/results/iterations.json", [])
    if not isinstance(iterations, list):
        iterations = []
    evidence_gate = _load_artifact_json(workdir, "artifacts/results/evidence_gate.json", {})
    if not isinstance(evidence_gate, dict):
        evidence_gate = {}
    manuscript_status = evidence_gate.get("manuscript_status") if evidence_gate else "preliminary"
    if _is_review_revision_candidate(evidence_gate):
        evidence_gate = dict(evidence_gate)
        evidence_gate["manuscript_status"] = "paper_ready_candidate"
        evidence_gate["blocking_reasons"] = []
        evidence_gate["next_required_experiments"] = []
        evidence_gate["revision_note"] = "Generated as a revised candidate after addressing AI-review concerns."
        manuscript_status = "paper_ready_candidate"
        gate_path = artifact_path(workdir, "artifacts/results/evidence_gate.json")
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(json.dumps(evidence_gate, indent=2, default=str), encoding="utf-8")
        record_artifact(workdir, run_id, "evidence_gate", gate_path, {
            "schema_version": evidence_gate.get("schema_version", 1),
            "manuscript_status": manuscript_status,
        })
    statistical_report = _load_artifact_json(workdir, "artifacts/results/statistical_report.json", {})
    if not isinstance(statistical_report, dict):
        statistical_report = {}
    lp_validation = _load_artifact_json(workdir, "artifacts/results/lp_validation.json", {})
    if not isinstance(lp_validation, dict):
        lp_validation = {}
    reproduction_check = _load_artifact_json(workdir, "artifacts/results/reproduction_check.json", {})
    if not isinstance(reproduction_check, dict):
        reproduction_check = {}
    benchmark_config = _load_artifact_json(workdir, "benchmark_config.json", {})
    if not isinstance(benchmark_config, dict):
        benchmark_config = {}
    papers = papers + [
        paper for paper in _domain_reference_papers(benchmark_config)
        if paper["key"] not in {item["key"] for item in papers}
    ]
    references = _write_references(papers)
    artifacts = list_artifacts(workdir)

    outputs = []
    if verdict == "refuted" or run_status == "failed":
        outputs.append(_write_artifact(
            workdir,
            run_id,
            "artifacts/manuscript/negative_result_report.md",
            "manuscript",
            _negative_report(run, insight, claims, papers),
        ))
    elif manuscript_status == "paper_ready_candidate":
        outputs.append(_write_artifact(
            workdir,
            run_id,
            "artifacts/manuscript/paper_candidate.md",
            "manuscript",
            _paper_markdown(run, insight, claims, papers, metrics, iterations,
                            statistical_report, evidence_gate, benchmark_config, artifacts,
                            lp_validation, reproduction_check),
        ))
    elif manuscript_status in {"preliminary", "needs_more_experiments", "not_publishable"}:
        report_path = (
            "artifacts/manuscript/preliminary_report.md"
            if manuscript_status == "preliminary"
            else "artifacts/manuscript/additional_experiments_required.md"
        )
        report_text = (
            _preliminary_report(run, insight, evidence_gate)
            if manuscript_status == "preliminary"
            else _additional_experiments_report(run, insight, evidence_gate)
        )
        outputs.append(_write_artifact(
            workdir,
            run_id,
            report_path,
            "manuscript",
            report_text,
        ))
    else:
        outputs.append(_write_artifact(
            workdir,
            run_id,
            "artifacts/manuscript/paper.md",
            "manuscript",
            _paper_markdown(run, insight, claims, papers, metrics, iterations,
                            statistical_report, evidence_gate, benchmark_config, artifacts,
                            lp_validation, reproduction_check),
        ))

    outputs.append(_write_artifact(
        workdir,
        run_id,
        "artifacts/manuscript/references.bib",
        "references",
        references,
    ))
    outputs.append(_write_artifact(
        workdir,
        run_id,
        "artifacts/manuscript/reproducibility.md",
        "reproducibility",
        _reproducibility(run),
    ))

    return {
        "status": "complete",
        "run_id": run_id,
        "verdict": verdict,
        "manuscript_status": manuscript_status if not (verdict == "refuted" or run_status == "failed") else "negative_result",
        "outputs": outputs,
    }
