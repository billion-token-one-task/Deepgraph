"""Compile a passed preflight into a portable, revision-pinned runner bundle."""

from __future__ import annotations

import importlib.util
import ast
import json
import py_compile
import shutil
from pathlib import Path
from typing import Any, Mapping

from meta_harness.runner_capability import ExperimentRequirements, RunnerRegistry


class RunnerMaterializationError(RuntimeError):
    pass


def _json_object(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, json.JSONDecodeError) as exc:
        raise RunnerMaterializationError(f"{label}_invalid") from exc
    if not isinstance(parsed, dict):
        raise RunnerMaterializationError(f"{label}_invalid")
    return parsed


def _returns_hook_input_unchanged(node: ast.Return, input_arg: str) -> bool:
    value = node.value
    if isinstance(value, ast.Name) and value.id == input_arg:
        return True
    return bool(
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "str"
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == input_arg
    )


def _validate_candidate_adapter(path: Path, requirements: ExperimentRequirements) -> None:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise RunnerMaterializationError("candidate_adapter_syntax_invalid") from exc
    protected_columns = {
        str(requirements.dataset.field_mapping.get(role) or "")
        for role in ("target", "label", "relevance")
    } - {""}
    for node in ast.walk(tree):
        key = None
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            key = node.slice.value
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            key = node.args[0].value
        if str(key) in protected_columns:
            raise RunnerMaterializationError("candidate_adapter_reads_target")
    hook_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == requirements.candidate_hook
        ),
        None,
    )
    # The generic runners always supply the example and the frozen baseline
    # input.  A callable alone is insufficient: a one-argument hook otherwise
    # survives materialization and fails only after scarce GPU work begins.
    # Keep this static: candidate source is untrusted and must never be invoked
    # by the controller during admission.
    if hook_node is None:
        raise RunnerMaterializationError(
            f"candidate_hook_missing:{requirements.candidate_hook}"
        )
    positional = len(hook_node.args.posonlyargs) + len(hook_node.args.args)
    if (
        positional != 2
        or hook_node.args.vararg is not None
        or hook_node.args.kwonlyargs
        or hook_node.args.kwarg is not None
    ):
        raise RunnerMaterializationError("candidate_hook_signature_invalid")
    if hook_node.decorator_list:
        raise RunnerMaterializationError("candidate_hook_signature_invalid")
    positional_args = [*hook_node.args.posonlyargs, *hook_node.args.args]
    if len(positional_args) >= 2:
        baseline_arg = positional_args[1].arg
        returns = [
            node for node in ast.walk(hook_node) if isinstance(node, ast.Return)
        ]
        if returns and all(
            _returns_hook_input_unchanged(node, baseline_arg) for node in returns
        ):
            raise RunnerMaterializationError("candidate_adapter_identity")
    try:
        py_compile.compile(str(path), doraise=True)
        spec = importlib.util.spec_from_file_location(
            f"deepgraph_candidate_adapter_{id(path)}", path
        )
        if spec is None or spec.loader is None:
            raise RunnerMaterializationError("candidate_adapter_unloadable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except RunnerMaterializationError:
        raise
    except Exception as exc:
        raise RunnerMaterializationError(
            f"candidate_adapter_invalid:{type(exc).__name__}:{exc}"
        ) from exc
    if not str(getattr(module, "CANDIDATE_METHOD", "")).strip():
        raise RunnerMaterializationError("candidate_method_missing")
    if not callable(getattr(module, requirements.candidate_hook, None)):
        raise RunnerMaterializationError(
            f"candidate_hook_missing:{requirements.candidate_hook}"
        )


def materialize_runner_bundle(
    *,
    workdir: str | Path,
    preflight_row: Mapping[str, Any],
    candidate_adapter_source: str,
) -> dict[str, Any]:
    """Write only the adapter selected by a durable passed preflight.

    The copied package makes the runner portable to SSH/Colab workdirs without
    relying on the controller checkout. Candidate code is compiled and imported
    before any compute attempt can be admitted.
    """

    if str(preflight_row.get("status") or "") != "passed":
        raise RunnerMaterializationError("passed_preflight_required")
    requirements = ExperimentRequirements.from_dict(
        _json_object(preflight_row.get("requirements_json"), label="requirements")
    )
    adapter_id = str(preflight_row.get("adapter_id") or "")
    capability = next(
        (item for item in RunnerRegistry().all() if item.adapter_id == adapter_id),
        None,
    )
    if capability is None or capability.structural_blockers(requirements):
        raise RunnerMaterializationError("preflight_adapter_contract_mismatch")
    dataset_revision = str(preflight_row.get("dataset_revision") or "").strip()
    model_revision = str(preflight_row.get("model_revision") or "").strip()
    if not dataset_revision or not model_revision:
        raise RunnerMaterializationError("resolved_revisions_required")
    source = str(candidate_adapter_source or "").strip()
    if not source:
        raise RunnerMaterializationError("candidate_adapter_required")

    code_dir = Path(workdir) / "code"
    package_dir = code_dir / "meta_harness"
    runner_dir = package_dir / "runners"
    runner_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    for source_path, target_path in (
        (source_root / "__init__.py", package_dir / "__init__.py"),
        (source_root / "failure_policy.py", package_dir / "failure_policy.py"),
        (source_root / "runner_capability.py", package_dir / "runner_capability.py"),
        (source_root / "runner_contract.py", package_dir / "runner_contract.py"),
        (source_root / "runners" / "__init__.py", runner_dir / "__init__.py"),
        (
            source_root / "runners" / "generic_transformers.py",
            runner_dir / "generic_transformers.py",
        ),
    ):
        shutil.copy2(source_path, target_path)

    adapter_path = code_dir / "candidate_adapter.py"
    adapter_path.write_text(source.rstrip() + "\n", encoding="utf-8")
    _validate_candidate_adapter(adapter_path, requirements)

    config = {
        "schema_version": "materialized_runner_v1",
        "adapter_id": adapter_id,
        "adapter_version": str(preflight_row.get("adapter_version") or ""),
        "preflight_result_id": int(preflight_row.get("id") or 0),
        "resolved_dataset_revision": dataset_revision,
        "resolved_model_revision": model_revision,
        "requirements": requirements.to_dict(),
    }
    config_path = code_dir / "execution_requirements.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    train_path = code_dir / "train.py"
    train_path.write_text(
        "import sys\n"
        "from meta_harness.runners.generic_transformers import main\n\n"
        "if __name__ == '__main__':\n"
        "    args = sys.argv[1:] or [\n"
        "        '--config', 'execution_requirements.json',\n"
        "        '--candidate-adapter', 'candidate_adapter.py',\n"
        "        '--output-dir', '../results',\n"
        "    ]\n"
        "    raise SystemExit(main(args))\n",
        encoding="utf-8",
    )
    dependencies = tuple(dict.fromkeys(capability.dependencies + requirements.dependencies))
    requirements_path = code_dir / "requirements.txt"
    requirements_path.write_text("\n".join(dependencies) + "\n", encoding="utf-8")
    for path in (train_path, adapter_path, runner_dir / "generic_transformers.py"):
        py_compile.compile(str(path), doraise=True)
    return {
        "adapter_id": adapter_id,
        "candidate_hook": requirements.candidate_hook,
        "baseline_command": "python train.py",
        "metric_name": requirements.metric.name,
        "metric_direction": requirements.metric.direction,
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "artifact_contract": list(requirements.artifact_contract),
        "paths": {
            "train": str(train_path),
            "candidate_adapter": str(adapter_path),
            "config": str(config_path),
            "requirements": str(requirements_path),
        },
    }
