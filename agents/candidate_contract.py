"""Judge a generated proposal against the runner contract before it is stored.

Why this exists
---------------
Between 2026-08-25 and 08-27 the proposer produced 35 candidates for agendas
16/17/18. Two of them were executable. The other 33 did not fail because the
model invented a dataset that does not exist -- across the whole 347-row
preflight history a repository was genuinely absent exactly once. They failed
on things a local function can decide in microseconds: a metric outside the
runner's vocabulary, a ``field_mapping`` keyed by column names instead of
contract roles, a ``preferred_backends`` list copied from the schema example,
a repository id written in Hugging Face's legacy bare form.

Nothing checked any of that until preflight, which runs *after* the candidate
has been stored, counted against the agenda's candidate quota and charged a
whole proposal grant. The generator was never told what was wrong, so the next
candidate repeated it. This module closes that loop: it renders the same
judgements preflight renders, before the row is written, and turns them into
text the generator can act on.

Two design rules
----------------
*Model-agnostic.* Nothing here knows or cares which model wrote the plan. Every
fact it asserts is read at call time from :class:`RunnerRegistry` -- the same
object preflight consults -- so a runner added tomorrow changes the advice
without touching this file. Swapping the proposer is a routing change, and the
loop works with any model that can read an error and try again.

*No catalogue of its own.* The one piece of world knowledge needed -- what
``wikitext`` is called now that Hugging Face namespaces everything -- is asked
of Hugging Face at call time rather than baked into a table here. A table would
be wrong within months and would have to be maintained by hand; the registry of
record already answers with a redirect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from meta_harness.runner_capability import (
    CapabilityContractError,
    ExperimentRequirements,
    HuggingFaceMetadataProbe,
    MetadataProbe,
    PreflightEnvironment,
    PreflightEngine,
    RepositoryMetadata,
    RunnerCapability,
    RunnerRegistry,
    apply_measurement_floors,
    fold_field_mapping_roles,
    validate_explicit_requirements_alignment,
)


# Conditions of the deployment, not defects in the plan. Preflight defers on
# these and a later pass retries them; no rewrite of the proposal can help, so
# regenerating against them would burn the candidate's attempts on weather.
ENVIRONMENT_CODES = frozenset(
    {
        "backend_unavailable",
        "network_unavailable",
        "disk_insufficient",
        "dependency_missing",
        "dataset_revision_unresolved",
        "model_revision_unresolved",
    }
)


@dataclass(frozen=True)
class ContractViolation:
    """One thing the generator must change, and what to change it to."""

    code: str
    detail: str

    @property
    def actionable(self) -> bool:
        """Whether rewriting the plan could remove this."""
        return self.code not in ENVIRONMENT_CODES

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True)
class ContractReview:
    """The verdict on one generated plan."""

    violations: tuple[ContractViolation, ...]
    plan: Mapping[str, Any]
    normalizations: tuple[str, ...] = ()
    remote_checked: bool = False

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(item.code for item in self.violations)

    @property
    def actionable(self) -> tuple[ContractViolation, ...]:
        """The subset a regenerated plan could actually repair."""
        return tuple(item for item in self.violations if item.actionable)


class RepositoryResolver:
    """Resolve repository identity against the registry of record, with a cache.

    Hugging Face answers a legacy bare id with a redirect to its namespaced
    home (``wikitext`` -> ``Salesforce/wikitext``), so canonicalisation and
    existence are the same question and cost one request. Results are cached
    for the life of the process because a proposal pass asks about the same
    dozen repositories repeatedly.

    Every method degrades to "no opinion" when the network is unavailable. A
    proposer that cannot reach Hugging Face must still be able to propose;
    turning an outage into a candidate refusal would trade one silent stall for
    another.
    """

    def __init__(self, *, probe: MetadataProbe | None = None):
        self._probe = probe or HuggingFaceMetadataProbe()
        self._canonical: dict[tuple[str, str], str] = {}
        self._metadata: dict[tuple[str, str, str, str], RepositoryMetadata] = {}

    # -- identity ---------------------------------------------------------
    def canonical_id(self, kind: str, repository_id: str) -> str:
        """Return the namespaced id Hugging Face redirects to, or the input."""
        raw = str(repository_id or "").strip()
        if not raw:
            return raw
        key = (kind, raw)
        if key in self._canonical:
            return self._canonical[key]
        resolved = self._ask_canonical(kind, raw)
        self._canonical[key] = resolved
        return resolved

    def _ask_canonical(self, kind: str, repository_id: str) -> str:
        import json as _json
        import urllib.error
        import urllib.parse
        import urllib.request

        segment = "datasets" if kind == "dataset" else "models"
        quoted = urllib.parse.quote(repository_id, safe="/")
        request = urllib.request.Request(
            f"https://huggingface.co/api/{segment}/{quoted}",
            headers={
                "Accept": "application/json",
                "User-Agent": "deepgraph-candidate-contract-v1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = _json.loads(response.read().decode("utf-8"))
        except Exception:
            return repository_id
        if not isinstance(payload, dict):
            return repository_id
        return str(payload.get("id") or repository_id)

    # -- metadata ---------------------------------------------------------
    def dataset(
        self, repository_id: str, revision: str, config: str
    ) -> RepositoryMetadata:
        key = ("dataset", repository_id, revision, config)
        if key not in self._metadata:
            self._metadata[key] = self._probe.dataset(repository_id, revision, config)
        return self._metadata[key]

    def model(self, repository_id: str, revision: str) -> RepositoryMetadata:
        key = ("model", repository_id, revision, "")
        if key not in self._metadata:
            self._metadata[key] = self._probe.model(repository_id, revision)
        return self._metadata[key]

    def dependency_available(self, name: str) -> bool:
        return self._probe.dependency_available(name)


# ---------------------------------------------------------------------------
# identity normalisation
# ---------------------------------------------------------------------------

_BARE_ID = re.compile(r"^[A-Za-z0-9][\w.\-]*$")


def _canonicalise_identities(
    plan: Mapping[str, Any], resolver: RepositoryResolver
) -> tuple[dict[str, Any], list[str]]:
    """Rewrite legacy bare repository ids to the namespaced form, everywhere.

    A bare id is not a hallucination: ``climate_fever`` was the dataset's real
    name until Hugging Face namespaced the hub, and the hub still answers to
    it. Refusing the plan and asking the model to guess the new owner wastes an
    attempt on a fact the hub will state for free. Only ids the hub actually
    redirects are rewritten; anything it does not recognise is left alone for
    the shape check to refuse.
    """

    plan = dict(plan)
    notes: list[str] = []
    requirements = dict(plan.get("execution_requirements") or {})
    if not requirements:
        return plan, notes

    rewrites: dict[tuple[str, str], str] = {}

    def _resolve(kind: str, value: Any) -> str:
        raw = str(value or "").strip()
        if not raw or not _BARE_ID.match(raw):
            return raw
        canonical = resolver.canonical_id(kind, raw)
        if canonical and canonical != raw and "/" in canonical:
            rewrites[(kind, raw)] = canonical
            notes.append(f"{kind} repository id {raw!r} resolved to {canonical!r}")
            return canonical
        return raw

    dataset = dict(requirements.get("dataset") or {})
    if dataset:
        dataset["repository_id"] = _resolve("dataset", dataset.get("repository_id"))
        requirements["dataset"] = dataset
    model = dict(requirements.get("model") or {})
    if model:
        model["repository_id"] = _resolve("model", model.get("repository_id"))
        requirements["model"] = model
    plan["execution_requirements"] = requirements

    # The identity guard binds the contract to the plan's own prose, so the
    # prose has to move with it or a rewrite here would manufacture a mismatch.
    if rewrites:
        plan["datasets"] = _rewrite_entries(plan.get("datasets"), "name", "dataset", rewrites)
        plan["benchmark_targets"] = _rewrite_entries(
            plan.get("benchmark_targets"), "hf_dataset", "dataset", rewrites
        )
        plan["baselines"] = _rewrite_entries(plan.get("baselines"), "model", "model", rewrites)
        plan["model_targets"] = _rewrite_entries(
            plan.get("model_targets"), "hf_model", "model", rewrites
        )
    return plan, notes


def _rewrite_entries(
    entries: Any, key: str, kind: str, rewrites: Mapping[tuple[str, str], str]
) -> Any:
    if not isinstance(entries, list):
        return entries
    updated = []
    for item in entries:
        if isinstance(item, Mapping):
            item = dict(item)
            raw = str(item.get(key) or "").strip()
            replacement = rewrites.get((kind, raw))
            if replacement:
                item[key] = replacement
        updated.append(item)
    return updated


# ---------------------------------------------------------------------------
# the review itself
# ---------------------------------------------------------------------------


def review_candidate_plan(
    plan: Mapping[str, Any],
    *,
    agenda: Any = None,
    environment: PreflightEnvironment | None = None,
    resolver: RepositoryResolver | None = None,
    registry: RunnerRegistry | None = None,
    check_remote: bool = True,
) -> ContractReview:
    """Return every reason this plan could not be executed as written.

    The offline stages mirror preflight's own order, so a plan this function
    passes is a plan preflight passes for the same reasons. The remote stage is
    literally :class:`PreflightEngine`, not a second implementation of it.
    """

    registry = registry or RunnerRegistry()
    resolver = resolver or RepositoryResolver()

    explicit = plan.get("execution_requirements")
    if not isinstance(explicit, Mapping) or not explicit:
        return ContractReview(
            (
                ContractViolation(
                    "candidate_execution_requirements_missing",
                    "the plan carries no execution_requirements block; "
                    "the runner contract is not optional",
                ),
            ),
            plan,
        )

    plan, normalizations = _canonicalise_identities(plan, resolver)
    requirements = apply_measurement_floors(
        fold_field_mapping_roles(
            ExperimentRequirements.parse_unvalidated(plan["execution_requirements"])
        )
    )
    capability = _nearest_capability(registry, requirements)
    violations: list[ContractViolation] = []

    try:
        requirements.validate()
    except CapabilityContractError as exc:
        violations.append(_describe(str(exc), requirements, capability, plan, registry))

    try:
        validate_explicit_requirements_alignment(plan, requirements)
    except CapabilityContractError as exc:
        violations.append(_describe(str(exc), requirements, capability, plan, registry))

    # ``RunnerRegistry.matches`` validates first, which would raise on the
    # leniently parsed object this function deliberately builds. Asking the
    # nearest capability directly is the same question without that
    # precondition: no blockers on the nearest means no blockers anywhere.
    for code in sorted(capability.structural_blockers(requirements)):
        violations.append(_describe(code, requirements, capability, plan, registry))

    violations.extend(_agenda_scope_violations(plan, agenda))

    if violations or not check_remote:
        return ContractReview(tuple(violations), plan, tuple(normalizations))

    environment = environment or _environment()
    result = PreflightEngine(registry=registry, probe=resolver).run(
        requirements, environment
    )
    for code in result.reason_codes:
        violations.append(
            _describe(code, requirements, capability, plan, registry, checks=result.checks)
        )
    return ContractReview(tuple(violations), plan, tuple(normalizations), True)


def _environment() -> PreflightEnvironment:
    from meta_harness.preflight_repository import runtime_preflight_environment

    return runtime_preflight_environment()


def _nearest_capability(
    registry: RunnerRegistry, requirements: ExperimentRequirements
) -> RunnerCapability:
    """The adapter preflight would report against: fewest blockers wins."""
    return sorted(
        registry.all(),
        key=lambda cap: (len(cap.structural_blockers(requirements)), cap.adapter_id),
    )[0]


def _agenda_scope_violations(plan: Mapping[str, Any], agenda: Any) -> list[ContractViolation]:
    """Run the topic gate's keyword rule here, where a rewrite is still free.

    The gate is deterministic and runs on the stored row, so a candidate that
    trips it is parked forever without ever being told which phrase did it --
    five candidates were re-refused 642 times in one day on 2026-08-27. Asking
    the same question before the row exists costs nothing and gives the
    generator the one fact it needs. The gate itself is untouched: this reads
    the agenda's own reject list and reports, it does not decide.
    """
    if agenda is None:
        return []
    try:
        phrases = [str(value) for value in (agenda.reject or {}).get("keywords") or []]
    except Exception:
        return []
    if not phrases:
        return []
    text = " ".join(
        str(plan.get(field) or "")
        for field in ("problem_statement", "proposed_method", "experimental_plan")
    ).lower()
    if not text.strip():
        import json as _json

        text = _json.dumps(plan, ensure_ascii=False, default=str).lower()
    found = sorted({phrase for phrase in phrases if phrase.lower() in text})
    if not found:
        return []
    return [
        ContractViolation(
            "topic_gate_agenda_reject_keyword",
            "the agenda refuses any candidate whose text contains "
            f"{found!r}. The gate matches the phrase, not the meaning, so a "
            "sentence that only says the method avoids it is refused too. "
            "State the property positively -- 'with frozen weights', "
            "'inference only' -- and do not use these phrases anywhere in the "
            "problem statement, method or plan.",
        )
    ]


def _join(values: Iterable[str]) -> str:
    return ", ".join(str(value) for value in values) or "(none)"


def _role_example(roles: Iterable[str]) -> str:
    return "{" + ", ".join(f'"{role}": "<dataset column>"' for role in roles) + "}"


def _plan_names(plan: Mapping[str, Any], field: str, key: str) -> list[str]:
    entries = plan.get(field)
    if not isinstance(entries, list):
        return []
    return [
        str(item.get(key))
        for item in entries
        if isinstance(item, Mapping) and item.get(key)
    ]


def _describe(
    code: str,
    requirements: ExperimentRequirements,
    capability: RunnerCapability,
    plan: Mapping[str, Any],
    registry: RunnerRegistry,
    *,
    checks: Mapping[str, Any] | None = None,
) -> ContractViolation:
    """Turn a preflight reason code into an instruction, read from the registry."""

    checks = checks or {}
    protocols = sorted({p for cap in registry.all() for p in cap.task_protocols})
    metrics = sorted({m for cap in registry.all() for m in cap.metric_names})
    hook = {
        "generative_qa": "candidate_prompt",
        "sequence_classification": "candidate_text",
    }.get(requirements.task_protocol)

    details: dict[str, str] = {
        "unsupported_task_protocol": (
            f"task_protocol {requirements.task_protocol!r} has no runner. "
            f"The registered protocols are: {_join(protocols)}."
        ),
        "candidate_hook_contract_invalid": (
            f"task_protocol {requirements.task_protocol!r} requires "
            f"candidate_hook {hook!r}; you wrote {requirements.candidate_hook!r}."
        ),
        "candidate_hook_unsupported": (
            f"candidate_hook {requirements.candidate_hook!r} is not one the "
            f"{capability.adapter_id} runner exposes: "
            f"{_join(capability.candidate_hooks)}."
        ),
        "dataset_schema_role_mismatch": (
            "dataset.field_mapping is keyed by CONTRACT ROLE, not by dataset "
            f"column. For task_protocol {requirements.task_protocol!r} the "
            f"required keys are exactly {_join(capability.dataset_roles)}; you "
            f"wrote {_join(sorted(requirements.dataset.field_mapping))}. "
            f"Write it as {_role_example(capability.dataset_roles)}."
        ),
        "model_framework_mismatch": (
            f"model.framework {requirements.model.framework!r} cannot be loaded "
            f"by the {capability.adapter_id} runner, which accepts "
            f"{_join(capability.model_frameworks)}."
        ),
        "model_task_mismatch": (
            f"model.task {requirements.model.task!r} is outside "
            f"{_join(capability.model_tasks)}"
            + (
                f"; the hub reports the checkpoint's published task as "
                f"{str(checks.get('model_task'))!r}"
                if checks.get("model_task")
                else ""
            )
            + ". No runner trains, so name a checkpoint that already publishes "
            "the head for the task you declare, or declare the task it has."
        ),
        "metric_contract_unsupported": (
            f"metric.name {requirements.metric.name!r} is not computable by any "
            f"runner. For task_protocol {requirements.task_protocol!r} the only "
            f"names are {_join(capability.metric_names)} (across all runners: "
            f"{_join(metrics)}). Pick one of those as the primary metric and "
            "move your bespoke quantity to a secondary observation, or choose a "
            "protocol whose vocabulary contains it."
        ),
        "backend_contract_mismatch": (
            f"preferred_backends {_join(requirements.preferred_backends)} "
            f"matches nothing the {capability.adapter_id} runner runs on: "
            f"{_join(capability.backends)}. Declare backends from that list."
        ),
        "seed_control_unsupported": (
            f"the {capability.adapter_id} runner evaluates one seed; declare a "
            "single seed."
        ),
        "sample_cap_unsupported": (
            f"the {capability.adapter_id} runner does not accept a sample cap."
        ),
        "artifact_contract_mismatch": (
            f"artifact_contract asks for artifacts the runner does not emit; it "
            f"emits {_join(capability.output_artifacts)}."
        ),
        "dataset_repository_id_malformed": (
            f"dataset.repository_id {requirements.dataset.repository_id!r} is "
            "not a Hugging Face repository id and the hub does not redirect it. "
            "Use the namespaced form 'owner/name'."
        ),
        "model_repository_id_malformed": (
            f"model.repository_id {requirements.model.repository_id!r} is not a "
            "Hugging Face repository id and the hub does not redirect it. Use "
            "the namespaced form 'owner/name'."
        ),
        "dataset_field_mapping_required": (
            "dataset.field_mapping is empty. It must map every contract role "
            f"{_join(capability.dataset_roles)} to a real dataset column."
        ),
        "execution_dataset_identity_unbound": (
            "the plan states no dataset repository id, so the contract is not "
            "bound to the science. Set 'name' on every datasets[] entry to the "
            "Hugging Face repository id ('owner/name') -- byte-for-byte the "
            f"same string as execution_requirements.dataset.repository_id "
            f"({requirements.dataset.repository_id!r}). You wrote: "
            f"{_join(_plan_names(plan, 'datasets', 'name'))}."
        ),
        "execution_dataset_identity_mismatch": (
            "execution_requirements.dataset.repository_id is "
            f"{requirements.dataset.repository_id!r} but the plan's datasets[] "
            f"name {_join(_plan_names(plan, 'datasets', 'name'))}. The contract "
            "must test the dataset the plan argues about; make them identical."
        ),
        "execution_model_identity_unbound": (
            "the plan states no model repository id. Set 'model' on every "
            "baselines[] entry to the Hugging Face repository id "
            "('owner/name'), one of them identical to "
            f"execution_requirements.model.repository_id "
            f"({requirements.model.repository_id!r})."
        ),
        "execution_model_identity_mismatch": (
            "execution_requirements.model.repository_id is "
            f"{requirements.model.repository_id!r} but the plan's baselines[] "
            f"name {_join(_plan_names(plan, 'baselines', 'model'))}. The "
            "measured model must be one the plan actually argues about."
        ),
        "execution_metric_identity_mismatch": (
            "metrics.primary names a different measurement from "
            f"execution_requirements.metric.name ({requirements.metric.name!r}). "
            f"Begin metrics.primary with the exact token "
            f"{requirements.metric.name!r} before any explanation."
        ),
        "dataset_unavailable": (
            f"the hub has no dataset {requirements.dataset.repository_id!r} at "
            f"revision {requirements.dataset.revision!r}. Name one that exists."
        ),
        "dataset_revision_unresolved": (
            f"the hub could not resolve revision "
            f"{requirements.dataset.revision!r} for "
            f"{requirements.dataset.repository_id!r}. Use 'main'."
        ),
        "dataset_schema_mismatch": (
            "the columns named in dataset.field_mapping do not exist in the "
            f"dataset: {_join(checks.get('dataset_missing_fields') or [])}. The "
            f"dataset publishes {_join(checks.get('dataset_fields') or [])}."
        ),
        "model_unavailable": (
            f"the hub has no model {requirements.model.repository_id!r} at "
            f"revision {requirements.model.revision!r}. Name one that exists."
        ),
        "model_revision_unresolved": (
            f"the hub could not resolve revision {requirements.model.revision!r} "
            f"for {requirements.model.repository_id!r}. Use 'main'."
        ),
        "dataset_schema_unverified": (
            f"the hub publishes no column list for "
            f"{requirements.dataset.repository_id!r}, so dataset.field_mapping "
            "cannot be verified before spending a grant. Name a dataset whose "
            "card declares its features."
        ),
        "vram_insufficient": (
            "the declared model does not fit the largest verified accelerator "
            f"({_join(str(round(float(value), 1)) for value in (checks.get('vram_required_gb'),) if value is not None)} GB "
            "required). Name a smaller checkpoint."
        ),
        "backend_unavailable": (
            "no enabled backend can run this contract right now. This is a "
            "deployment condition, not a defect in the plan."
        ),
        "network_unavailable": (
            "the controller has no network. A deployment condition, not a "
            "defect in the plan."
        ),
        "dependency_missing": (
            "a runtime dependency the runner needs is absent on the "
            "controller. A deployment condition, not a defect in the plan."
        ),
        "disk_insufficient": (
            "the declared model is larger than the free disk on the controller."
        ),
    }
    return ContractViolation(code, details.get(code, f"preflight refuses this plan ({code})."))


def render_violations(review: ContractReview) -> str:
    """The block appended to the design prompt before the next attempt."""
    lines = [
        "## YOUR PREVIOUS PLAN WAS REFUSED - fix EVERY item below",
        "",
        "These are not stylistic notes. Each one is a check the execution",
        "gate runs on the plan you just returned, and it refused it. Return",
        "the same JSON object with all of them repaired at once; fixing only",
        "some of them fails again for the remainder.",
        "",
    ]
    for index, violation in enumerate(review.violations, start=1):
        lines.append(f"{index}. [{violation.code}] {violation.detail}")
    if review.normalizations:
        lines.append("")
        lines.append(
            "Already corrected for you (keep these values): "
            + "; ".join(review.normalizations)
        )
    return "\n".join(lines)
