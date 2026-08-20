"""Generic real-data/real-model runner for two structured task protocols."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from meta_harness.failure_policy import classify_failure
from meta_harness.runner_capability import ExperimentRequirements
from meta_harness.runner_contract import (
    ResearchRunner,
    RunnerContractError,
    paired_permutation_test,
    recompute_metric,
    validate_final_results,
)


def _dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_input_sha256(value: str) -> str:
    normalized = " ".join(str(value).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _clamped_seeds(seeds: Sequence[int]) -> list[int]:
    """Honor DEEPGRAPH_RUNNER_MAX_SEEDS as a prefix cap on the seed list.

    Set only on pilot-stage compute requests. Zero or absent means the full
    declared list; the clamp keeps list order so one seed always means the
    design's first seed, and manifests report whatever actually ran.
    """
    declared = [int(seed) for seed in seeds]
    try:
        cap = int(os.environ.get("DEEPGRAPH_RUNNER_MAX_SEEDS", "0") or 0)
    except ValueError:
        cap = 0
    if cap > 0:
        return declared[:cap]
    return declared


def _load_candidate(path: Path, protocol: str):
    if not path.is_file():
        raise RunnerContractError("runner_contract_violation", "candidate_adapter_missing")
    spec = importlib.util.spec_from_file_location("deepgraph_candidate_adapter", path)
    if spec is None or spec.loader is None:
        raise RunnerContractError("runner_contract_violation", "candidate_adapter_unloadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    method_name = str(getattr(module, "CANDIDATE_METHOD", "")).strip()
    if not method_name:
        raise RunnerContractError("runner_contract_violation", "candidate_method_missing")
    hook = "candidate_prompt" if protocol == "generative_qa" else "candidate_text"
    if not callable(getattr(module, hook, None)):
        raise RunnerContractError("runner_contract_violation", f"{hook}_missing")
    return module, method_name, hook


class GenericTransformersRunner(ResearchRunner):
    """One model and dataset revision, paired baseline/candidate evaluation."""

    BASELINE_METHOD = "unmodified_input_baseline"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        candidate_adapter_path: str | Path,
        output_dir: str | Path,
    ):
        requirement_payload = config.get("requirements") or config
        self.requirements = ExperimentRequirements.from_dict(requirement_payload)
        # Stage policy, not science: a pilot exists to prove the measurement
        # works, and one seed at full n already does that (run 164's pilot
        # was accepted at one seed). The full benchmark and the audit holdout
        # never set this env, so every scientific claim still carries the
        # design's complete seed list, recorded truthfully in the manifests.
        self.seeds = _clamped_seeds(self.requirements.seeds)
        self.config = dict(config)
        self.dataset_revision = str(
            config.get("resolved_dataset_revision")
            or self.requirements.dataset.revision
        )
        self.model_revision = str(
            config.get("resolved_model_revision")
            or self.requirements.model.revision
        )
        self.output_dir = Path(output_dir)
        self.candidate_path = Path(candidate_adapter_path)
        self.candidate_module = None
        self.candidate_method = ""
        self.candidate_hook = ""
        self.dataset_rows: list[dict[str, Any]] = []
        self.tokenizer = None
        self.model = None
        self.torch = None
        self.predictions: list[dict[str, Any]] = []
        self.metrics: dict[str, float] = {}
        self.significance: dict[str, Any] = {}
        self.started_at = time.time()

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_module, self.candidate_method, self.candidate_hook = _load_candidate(
            self.candidate_path,
            self.requirements.task_protocol,
        )
        try:
            import torch
        except ImportError as exc:
            raise RunnerContractError("dependency_missing", "torch") from exc
        self.torch = torch
        for seed in self.seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def load_dataset(self) -> Sequence[Mapping[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RunnerContractError("dependency_missing", "datasets") from exc
        dataset_args = [self.requirements.dataset.repository_id]
        if self.requirements.dataset.config:
            dataset_args.append(self.requirements.dataset.config)
        try:
            dataset = load_dataset(
                *dataset_args,
                split=self.requirements.dataset.split,
                revision=self.dataset_revision,
            )
        except Exception as exc:
            reason = classify_failure(message=f"dataset unavailable:{exc}")
            raise RunnerContractError(reason, str(exc)) from exc
        cap = self.requirements.sample_cap or len(dataset)
        # A holdout audit evaluates examples the original run never saw.
        # The offset is execution addressing, not scientific identity: the
        # sample window is recorded per-row via sample_index and in the
        # dataset manifest, so the evidence trail states exactly which
        # examples were measured.
        offset = int(
            dict(self.config.get("runtime_adjustments") or {}).get("example_offset")
            or self.config.get("example_offset")
            or os.environ.get("DEEPGRAPH_RUNNER_EXAMPLE_OFFSET")
            or 0
        )
        start = max(0, min(offset, len(dataset)))
        end = min(len(dataset), start + cap)
        self.dataset_rows = [dict(dataset[index]) for index in range(start, end)]
        self.example_offset = start
        if not self.dataset_rows:
            raise RunnerContractError("dataset_unavailable", "empty_split")
        missing = sorted(
            set(self.requirements.dataset.field_mapping.values())
            - set(self.dataset_rows[0])
        )
        if missing:
            raise RunnerContractError("dataset_schema_mismatch", ",".join(missing))
        return self.dataset_rows

    def load_model(self) -> Any:
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RunnerContractError("dependency_missing", "transformers") from exc
        kwargs: dict[str, Any] = {"revision": self.model_revision}
        if self.torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = "auto"
        runtime_adjustments = dict(self.config.get("runtime_adjustments") or {})
        use_4bit = self.requirements.model.quantization == "4bit" or bool(
            runtime_adjustments.get("prefer_quantized")
        )
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except ImportError as exc:
                raise RunnerContractError("dependency_missing", "bitsandbytes") from exc
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.requirements.model.repository_id,
                revision=self.model_revision,
            )
            model_class = (
                AutoModelForCausalLM
                if self.requirements.task_protocol == "generative_qa"
                else AutoModelForSequenceClassification
            )
            self.model = model_class.from_pretrained(
                self.requirements.model.repository_id,
                **kwargs,
            )
            self.model.eval()
        except Exception as exc:
            reason = classify_failure(message=f"model load:{exc}")
            raise RunnerContractError(reason, str(exc)) from exc
        return self.model

    def _device(self):
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")

    def _qa_prediction(self, prompt: str) -> tuple[str, bool]:
        """Return the continuation and whether generation ran out of budget."""
        return self._qa_predictions([prompt])[0]

    def _qa_predictions(self, prompts: list[str]) -> list[tuple[str, bool]]:
        """Batched greedy generation; one (text, truncated) tuple per prompt.

        A generation that stops on the token cap did not answer the question,
        it was interrupted. Run 153 spent its whole grant with all 24
        predictions cut off mid-sentence under the 64-token default, so
        exact_match was zero by construction and the result was filed as a
        refutation. Batching exists for the same honesty reason from the other
        side: single-stream decoding of a floor-compliant run (200 examples,
        two arms, three seeds, 512 tokens) needs ~8-12 GPU-hours on a T4 and
        would burn through the pilot grant's compute cap mid-measurement.
        """
        # _qa_prediction stays the override point: harness fakes and any
        # subclass that stubs single-prompt decoding keep working unchanged.
        if type(self)._qa_prediction is not GenericTransformersRunner._qa_prediction:
            return [self._qa_prediction(prompt) for prompt in prompts]
        runtime_adjustments = dict(self.config.get("runtime_adjustments") or {})
        # Default measured, not guessed: the 2026-08-17 M0 probe on GSM8K
        # showed 59% of generations still hit a 256-token cap, and truncated
        # samples scored 9% against 61% for completed ones. 512 is the V1
        # scaffold value (docs/internal/V1_SCAFFOLD_REGISTER.md).
        max_new_tokens = int(
            runtime_adjustments.get("max_new_tokens")
            or self.config.get("max_new_tokens")
            or 512
        )
        batch_size = int(
            runtime_adjustments.get("generation_batch_size")
            or self.config.get("generation_batch_size")
            or os.environ.get("DEEPGRAPH_RUNNER_BATCH_SIZE")
            or 16
        )
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        eos_id = self.tokenizer.eos_token_id
        # Decoder-only batch generation must left-pad or continuations start
        # from pad positions.
        previous_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        results: list[tuple[str, bool]] = []
        try:
            for start in range(0, len(prompts), max(1, batch_size)):
                chunk = prompts[start:start + max(1, batch_size)]
                tokens = self.tokenizer(chunk, return_tensors="pt", padding=True)
                tokens = {
                    key: value.to(self._device()) for key, value in tokens.items()
                }
                with self.torch.inference_mode():
                    generated = self.model.generate(
                        **tokens,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=pad_id,
                    )
                prompt_length = tokens["input_ids"].shape[1]
                for row in range(generated.shape[0]):
                    continuation = generated[row][prompt_length:].tolist()
                    while continuation and pad_id is not None and continuation[-1] == pad_id:
                        continuation.pop()
                    stopped_on_eos = bool(
                        continuation
                        and eos_id is not None
                        and int(continuation[-1]) == int(eos_id)
                    )
                    # pad == eos on many chat models: the terminating eos is
                    # stripped with the padding, so any sequence shorter than
                    # the cap stopped on its own.
                    truncated = bool(
                        len(continuation) >= max_new_tokens and not stopped_on_eos
                    )
                    text = self.tokenizer.decode(
                        continuation, skip_special_tokens=True
                    ).strip()
                    results.append((text, truncated))
        finally:
            self.tokenizer.padding_side = previous_side
        return results

    def _classification_prediction(self, text: str) -> str:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
        tokens = {key: value.to(self._device()) for key, value in tokens.items()}
        with self.torch.inference_mode():
            logits = self.model(**tokens).logits
        return str(int(logits.argmax(dim=-1).item()))

    def _run_method(self, method: str, *, candidate: bool) -> list[dict[str, Any]]:
        mapping = self.requirements.dataset.field_mapping
        output: list[dict[str, Any]] = []
        for seed in self.seeds:
            random.seed(seed)
            self.torch.manual_seed(seed)
            if self.requirements.task_protocol == "generative_qa":
                model_inputs: list[str] = []
                targets: list[str] = []
                for example in self.dataset_rows:
                    baseline_input = str(example[mapping["prompt"]])
                    candidate_example = {
                        key: value
                        for key, value in example.items()
                        if key != mapping["target"]
                    }
                    model_inputs.append(
                        str(
                            self.candidate_module.candidate_prompt(
                                candidate_example, baseline_input
                            )
                        )
                        if candidate
                        else baseline_input
                    )
                    targets.append(str(example[mapping["target"]]))
                generations = self._qa_predictions(model_inputs)
                for index, (model_input, target, (prediction, truncated)) in enumerate(
                    zip(model_inputs, targets, generations)
                ):
                    output.append(
                        {
                            "method": method,
                            "seed": seed,
                            "sample_index": index,
                            "prediction": prediction,
                            # Recorded so a run that was interrupted cannot be
                            # read as a model that answered badly.
                            "truncated": bool(truncated),
                            "target": target,
                            "input_sha256": hashlib.sha256(
                                model_input.encode("utf-8")
                            ).hexdigest(),
                            "normalized_input_sha256": _normalized_input_sha256(
                                model_input
                            ),
                        }
                    )
                continue
            for index, example in enumerate(self.dataset_rows):
                baseline_input = str(example[mapping["text"]])
                candidate_example = {
                    key: value
                    for key, value in example.items()
                    if key != mapping["label"]
                }
                model_input = (
                    str(
                        self.candidate_module.candidate_text(
                            candidate_example, baseline_input
                        )
                    )
                    if candidate
                    else baseline_input
                )
                prediction = self._classification_prediction(model_input)
                truncated = False
                target = str(example[mapping["label"]])
                output.append(
                    {
                        "method": method,
                        "seed": seed,
                        "sample_index": index,
                        "prediction": prediction,
                        # Recorded so a run that was interrupted cannot be read as
                        # a model that answered badly.
                        "truncated": bool(truncated),
                        "target": target,
                        "input_sha256": hashlib.sha256(
                            model_input.encode("utf-8")
                        ).hexdigest(),
                        "normalized_input_sha256": _normalized_input_sha256(
                            model_input
                        ),
                    }
                )
        self.predictions.extend(output)
        return output

    def run_baseline(self) -> Sequence[Mapping[str, Any]]:
        return self._run_method(self.BASELINE_METHOD, candidate=False)

    def run_candidate(self) -> Sequence[Mapping[str, Any]]:
        candidate_rows = self._run_method(self.candidate_method, candidate=True)
        baseline_hashes = {
            (int(row["seed"]), int(row["sample_index"])): str(
                row["normalized_input_sha256"]
            )
            for row in self.predictions
            if row["method"] == self.BASELINE_METHOD
        }
        candidate_hashes = {
            (int(row["seed"]), int(row["sample_index"])): str(
                row["normalized_input_sha256"]
            )
            for row in candidate_rows
        }
        if set(candidate_hashes) != set(baseline_hashes):
            raise RunnerContractError(
                "runner_contract_violation", "candidate_pairing_mismatch"
            )
        if candidate_hashes and all(
            candidate_hashes[key] == baseline_hashes[key]
            for key in candidate_hashes
        ):
            raise RunnerContractError(
                "runner_contract_violation", "candidate_adapter_identity"
            )
        return candidate_rows

    def compute_metrics(self) -> Mapping[str, Any]:
        metric_name = self.requirements.metric.name
        baseline_rows = [
            row for row in self.predictions if row["method"] == self.BASELINE_METHOD
        ]
        candidate_rows = [
            row for row in self.predictions if row["method"] == self.candidate_method
        ]
        self.metrics = {
            self.BASELINE_METHOD: recompute_metric(baseline_rows, metric_name),
            self.candidate_method: recompute_metric(candidate_rows, metric_name),
        }
        # A difference without a significance test cannot become a supported
        # verdict: decide_evidence blocks on a missing p-value. The arms are
        # already paired by (seed, sample_index), so the test costs no extra
        # inference -- only arithmetic over predictions we have already made.
        self.significance = paired_permutation_test(
            baseline_rows,
            candidate_rows,
            metric_name,
            seed=int(self.seeds[0]) if self.seeds else 0,
        )
        return self.metrics

    def _gpu_environment(self) -> dict[str, Any]:
        available = bool(self.torch.cuda.is_available())
        return {
            "available": available,
            "device_count": int(self.torch.cuda.device_count()) if available else 0,
            "device_name": self.torch.cuda.get_device_name(0) if available else "cpu",
            "cuda_version": self.torch.version.cuda,
            "torch_version": self.torch.__version__,
            "python_version": platform.python_version(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }

    def emit_final_results(self) -> Mapping[str, Any]:
        if not self.metrics:
            raise RunnerContractError("metric_missing")
        if not self.significance:
            raise RunnerContractError("p_value_missing", "compute_metrics_not_run")
        # A run where nothing finished generating did not test the hypothesis;
        # it ran out of token budget. Run 153 emitted 24 truncated predictions
        # under the 64-token default and the zero-vs-zero result was filed as a
        # refutation.
        truncated = sum(1 for row in self.predictions if row.get("truncated"))
        if self.predictions and truncated == len(self.predictions):
            raise RunnerContractError(
                "generation_truncated",
                f"all {truncated} predictions stopped on the token cap",
            )
        raw_path = self.output_dir / "raw_predictions.jsonl"
        raw_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in self.predictions
            ),
            encoding="utf-8",
        )
        environment_path = self.output_dir / "environment_manifest.json"
        dataset_path = self.output_dir / "dataset_manifest.json"
        model_path = self.output_dir / "model_manifest.json"
        _dump(environment_path, self._gpu_environment())
        _dump(
            dataset_path,
            {
                "repository_id": self.requirements.dataset.repository_id,
                "revision": self.dataset_revision,
                "config": self.requirements.dataset.config,
                "split": self.requirements.dataset.split,
                "field_mapping": dict(self.requirements.dataset.field_mapping),
                "num_examples": len(self.dataset_rows),
                "example_offset": int(getattr(self, "example_offset", 0)),
            },
        )
        _dump(
            model_path,
            {
                "repository_id": self.requirements.model.repository_id,
                "revision": self.model_revision,
                "framework": self.requirements.model.framework,
                "task": self.requirements.model.task,
                "quantization": self.requirements.model.quantization,
            },
        )
        baseline = float(self.metrics[self.BASELINE_METHOD])
        candidate = float(self.metrics[self.candidate_method])
        direction = self.requirements.metric.direction
        negative = candidate <= baseline if direction == "higher" else candidate >= baseline
        # A verdict is a scientific claim, and "refuted" needs the same
        # evidential standard as "supported". Deriving it from the SIGN alone
        # called run 164 (delta -0.03, p=0.506) and run 180 (delta -0.06,
        # p=0.071) refuted -- two of the eight audited ladders overstated
        # their result, and the cross-vendor evaluator correctly dissented on
        # run 189 (p=0.220) for exactly this reason. Failing to show an
        # improvement is not the same as showing harm.
        _p = self.significance.get("paired_permutation_p")
        _significant = _p is not None and float(_p) < 0.05
        if not _significant:
            hypothesis_verdict = "inconclusive"
        elif negative:
            hypothesis_verdict = "refuted"
        else:
            hypothesis_verdict = "supported"
        artifacts = {
            "final_results": {"path": "final_results.json"},
            "raw_predictions": {"path": raw_path.name},
            "environment_manifest": {"path": environment_path.name},
            "dataset_manifest": {"path": dataset_path.name},
            "model_manifest": {"path": model_path.name},
        }
        hashes = {
            "raw_predictions": _sha256(raw_path),
            "environment_manifest": _sha256(environment_path),
            "dataset_manifest": _sha256(dataset_path),
            "model_manifest": _sha256(model_path),
            "candidate_adapter": _sha256(self.candidate_path),
        }
        result = {
            "schema_version": "final_results_v1",
            "task_protocol": self.requirements.task_protocol,
            "dataset_id": self.requirements.dataset.repository_id,
            "dataset_revision": self.dataset_revision,
            "model_id": self.requirements.model.repository_id,
            "model_revision": self.model_revision,
            "seeds": list(self.seeds),
            "num_seeds": len(self.seeds),
            "num_examples": len(self.dataset_rows),
            "baseline_method": self.BASELINE_METHOD,
            "candidate_method": self.candidate_method,
            "metric_name": self.requirements.metric.name,
            "primary_metric": self.requirements.metric.name,
            "metric_direction": direction,
            "metric_value": candidate,
            "baseline_metric_value": baseline,
            "best_metric_value": candidate,
            "statistical_tests": dict(self.significance),
            # The one authoritative verdict. Consumers must read this rather
            # than re-deriving from scientific_negative_result, which records
            # only the direction of the difference.
            "hypothesis_verdict": hypothesis_verdict,
            "per_method": {
                self.BASELINE_METHOD: {
                    self.requirements.metric.name: baseline,
                    "metric_value": baseline,
                },
                self.candidate_method: {
                    self.requirements.metric.name: candidate,
                    "metric_value": candidate,
                },
            },
            "seed_results": [
                {
                    "seed": seed,
                    "baseline": recompute_metric(
                        [
                            row
                            for row in self.predictions
                            if row["method"] == self.BASELINE_METHOD
                            and row["seed"] == seed
                        ],
                        self.requirements.metric.name,
                    ),
                    "candidate": recompute_metric(
                        [
                            row
                            for row in self.predictions
                            if row["method"] == self.candidate_method
                            and row["seed"] == seed
                        ],
                        self.requirements.metric.name,
                    ),
                }
                for seed in self.seeds
            ],
            "scientific_negative_result": negative,
            "execution_reason_code": (
                "scientific_negative_result" if negative else "attempt_completed"
            ),
            "label_fallback_used": False,
            "gpu_environment": self._gpu_environment(),
            "artifacts": artifacts,
            "artifact_hashes": hashes,
            "wall_seconds": time.time() - self.started_at,
        }
        validate_final_results(result)
        final_path = self.output_dir / "final_results.json"
        _dump(final_path, result)
        return result

    def run(self) -> Mapping[str, Any]:
        self.prepare()
        print("BENCHMARK_STAGE: prepare_complete", flush=True)
        self.load_dataset()
        print("BENCHMARK_STAGE: dataset_ready", flush=True)
        self.load_model()
        print("BENCHMARK_STAGE: model_ready", flush=True)
        self.run_baseline()
        self.run_candidate()
        self.compute_metrics()
        result = self.emit_final_results()
        print("FINAL_RESULTS: " + json.dumps(result, ensure_ascii=False), flush=True)
        return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        GenericTransformersRunner(
            config,
            candidate_adapter_path=args.candidate_adapter,
            output_dir=args.output_dir,
        ).run()
        return 0
    except Exception as exc:
        reason = (
            exc.reason_code
            if isinstance(exc, RunnerContractError)
            else classify_failure(message=str(exc), returncode=1)
        )
        print(
            "RUNNER_ERROR: "
            + json.dumps(
                {"reason_code": reason, "detail": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
