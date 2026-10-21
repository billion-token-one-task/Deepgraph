"""Per-sample inference against a hosted model, with no accelerator.

Why this exists: every runner before it loads weights onto a GPU, so every
experiment this system could admit had to be a prompt or scaffold intervention
on a local checkpoint -- and almost every failure in the ledger came from the
transport that delivered it, not from the science. Twenty-five ssh_gpu
dispatches failed and none succeeded; nine leases expired; a run today died
because a rented host arrived without `transformers` installed. None of that
is research. This runner deletes that whole class of failure for the kind of
experiment that never needed a GPU in the first place: send a prompt, read a
completion, score it.

It is deliberately a SUBCLASS rather than a sibling. Scoring, the paired
permutation test, the artifact contract, the manifests and the truncation
bookkeeping all stay in one implementation, so "the API path is held to the
same standard as the GPU path" is guaranteed by the code being the same code
rather than asserted in a document. What is overridden is exactly the part
that differs: where a prediction comes from.

What is knowingly weaker: provenance. A local run records the weights it used
by hash, and a hosted endpoint cannot be pinned that way -- the provider may
requantise or reroute without telling anyone. So the model manifest records
what CAN be checked (provider, model id, sampling parameters, the date, and a
digest over the returned text) and says plainly, in the artifact itself, that
this is endpoint-level and not weight-level reproducibility. A reader deserves
to know which of the two they are holding.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
import urllib.parse
import urllib.request
from typing import Any, Mapping, Sequence

from meta_harness.failure_policy import classify_failure
from meta_harness.runner_contract import RunnerContractError
from meta_harness.runners.generic_transformers import GenericTransformersRunner

# Where a bundle running off-host reads its rows from when the `datasets`
# package is absent. The controller has no reason to carry a heavy data
# dependency to fetch a couple of hundred rows over HTTP.
DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
MAX_ROWS_PER_REQUEST = 100


class ApiInferenceRunner(GenericTransformersRunner):
    """Same experiment, same scoring, predictions from an endpoint."""

    def prepare(self) -> None:
        # The parent seeds torch here and refuses to start without it. Nothing
        # in this path is torch-shaped, and requiring a CUDA stack to make an
        # HTTPS request is the dependency this runner exists to drop.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        from meta_harness.runners.generic_transformers import _load_candidate

        (
            self.candidate_module,
            self.candidate_method,
            self.candidate_hook,
        ) = _load_candidate(self.candidate_path, self.requirements.task_protocol)
        self.torch = _SeedOnly()
        for seed in self.seeds:
            random.seed(seed)
        self._call_count = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._response_digest = hashlib.sha256()
        self._served_models: set[str] = set()

    def load_dataset(self) -> Sequence[Mapping[str, Any]]:
        try:
            return super().load_dataset()
        except RunnerContractError as exc:
            if exc.reason_code != "dependency_missing":
                raise
        # Same window arithmetic as the parent, against the HTTP rows service.
        # Kept identical on purpose: the holdout's whole validity rests on the
        # offset meaning the same thing on both paths.
        cap = self.requirements.sample_cap or MAX_ROWS_PER_REQUEST
        offset = int(
            dict(self.config.get("runtime_adjustments") or {}).get("example_offset")
            or self.config.get("example_offset")
            or os.environ.get("DEEPGRAPH_RUNNER_EXAMPLE_OFFSET")
            or 0
        )
        rows, total = self._fetch_rows(offset, cap)
        self.dataset_rows = rows
        self.example_offset = offset
        self.split_total = total
        if not self.dataset_rows:
            raise RunnerContractError("dataset_unavailable", "empty_split")
        missing = sorted(
            set(self.requirements.dataset.field_mapping.values())
            - set(self.dataset_rows[0])
        )
        if missing:
            raise RunnerContractError("dataset_schema_mismatch", ",".join(missing))
        return self.dataset_rows

    def _fetch_rows(self, offset: int, cap: int) -> tuple[list[dict], int]:
        collected: list[dict] = []
        total = 0
        while len(collected) < cap:
            want = min(MAX_ROWS_PER_REQUEST, cap - len(collected))
            query = {
                "dataset": self.requirements.dataset.repository_id,
                "config": self.requirements.dataset.config or "default",
                "split": self.requirements.dataset.split,
                "offset": offset + len(collected),
                "length": want,
            }
            url = f"{DATASETS_SERVER}?{urllib.parse.urlencode(query)}"
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    payload = json.load(response)
            except Exception as exc:                                # noqa: BLE001
                reason = classify_failure(message=f"dataset unavailable:{exc}")
                raise RunnerContractError(reason, str(exc)) from exc
            batch = [item["row"] for item in payload.get("rows") or []]
            total = int(payload.get("num_rows_total") or total)
            if not batch:
                break
            collected.extend(batch)
        return collected[:cap], total or (offset + len(collected))

    def load_model(self) -> Any:
        """Resolve the endpoint. Nothing is downloaded and nothing is loaded."""
        model_id = str(self.requirements.model.repository_id or "").strip()
        if not model_id:
            raise RunnerContractError("model_unavailable", "no model id")
        self.provider_name = str(
            dict(self.config.get("runtime_adjustments") or {}).get("provider")
            or self.config.get("provider")
            or os.environ.get("DEEPGRAPH_API_RUNNER_PROVIDER")
            or ""
        ).strip()
        if not self.provider_name:
            raise RunnerContractError("model_unavailable", "no provider configured")
        try:
            from agents.llm_client import call_llm_with_provider
        except ImportError as exc:
            raise RunnerContractError("dependency_missing", "llm_client") from exc
        self._call = call_llm_with_provider
        self.model = f"{self.provider_name}:{model_id}"
        return self.model

    # ── predictions ──────────────────────────────────────────────────────
    # Greedy, one prompt per call, no chat history. The parent's batching
    # exists to stop single-stream decoding from blowing an hour of GPU on a
    # 200-example run; an endpoint has no such cost, and a batch would only
    # blur which completion answered which prompt.

    # No system prompt, for two reasons that point the same way. The local
    # runner sends the bare prompt and scores the continuation, so anything
    # added here would make the two paths measure different things -- and the
    # comparison between them is the reason this adapter is a subclass. It is
    # also what the gateway actually answers: an instruction to reply tersely
    # made sora2_gemini return empty content on every short question, while
    # the same question with no system message answered normally. The scorer
    # already reads a number out of a sentence, so terseness was never needed.
    _SYSTEM = ""

    def _one_call(self, prompt: str) -> tuple[str, bool]:
        # Resolved exactly as the local path resolves it, so a run that moves
        # between the two is not silently given a different answer budget --
        # truncation is the difference between a wrong answer and no answer,
        # and 59% of one earlier run's generations died on a 256-token cap.
        adjustments = dict(self.config.get("runtime_adjustments") or {})
        max_tokens = int(
            adjustments.get("max_new_tokens")
            or self.config.get("max_new_tokens")
            or 512
        )
        try:
            text, _tokens, provider = self._call(
                self._SYSTEM,
                prompt,
                provider_name=self.provider_name,
                max_tokens=max_tokens,
            )
        except Exception as exc:                                    # noqa: BLE001
            reason = classify_failure(message=f"inference:{exc}")
            raise RunnerContractError(reason, str(exc)) from exc
        answer = str(text or "")
        self._call_count += 1
        # What the endpoint says it served, not what the config asked for. A
        # gateway that silently reroutes to a different model is exactly the
        # provenance gap this path has instead of a weight hash, so the one
        # identity we CAN observe is recorded from the response side.
        served = str((provider or {}).get("model") or "")
        if served:
            self._served_models.add(served)
        self._response_digest.update(answer.encode("utf-8"))
        # A hosted model stops when it decides to, so "ran out of budget" is
        # not observable the way it is locally. Reporting it as never truncated
        # would be a claim we cannot support; reporting the length check we CAN
        # make keeps the field honest.
        truncated = len(answer) >= max_tokens * 4
        return answer, truncated

    def _qa_predictions(self, prompts: list[str]) -> list[tuple[str, bool]]:
        return [self._one_call(prompt) for prompt in prompts]

    def _classification_prediction(self, text: str) -> str:
        answer, _truncated = self._one_call(text)
        return answer.strip().splitlines()[0] if answer.strip() else ""

    def _device(self):
        return "api"

    def _gpu_environment(self) -> dict[str, Any]:
        """What ran this, stated for what it is.

        `available: false` is the literal truth -- no accelerator was involved
        -- and the reproducibility field says out loud that a hosted endpoint
        is pinned by name and date rather than by weight hash, so nobody reads
        this manifest as the stronger claim.
        """
        return {
            "available": False,
            "device_count": 0,
            "device_name": "hosted-api",
            "cuda_version": None,
            "torch_version": None,
            "python_version": platform.python_version(),
            "visible_devices": "",
            "inference": {
                "transport": "hosted_api",
                "provider": getattr(self, "provider_name", ""),
                "model_id": self.requirements.model.repository_id,
                "served_model_ids": sorted(self._served_models),
                "calls": self._call_count,
                "decoding": "greedy",
                "response_digest_sha256": self._response_digest.hexdigest(),
                "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "reproducibility": (
                    "endpoint-level: provider, model id, decoding and date are "
                    "recorded and the returned text is digested, but a hosted "
                    "endpoint cannot be pinned to a weight revision the way a "
                    "local checkpoint can"
                ),
            },
        }


class _SeedOnly:
    """Stands in for the torch handle the parent seeds and queries.

    The parent calls `torch.manual_seed` per method and `torch.cuda.*` when
    building the environment manifest. Those are the only two things it needs
    from torch, and an endpoint has neither, so the calls are absorbed rather
    than the whole dependency being dragged in for them.
    """

    class cuda:                                                     # noqa: N801
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

        @staticmethod
        def manual_seed_all(_seed: int) -> None:
            return None

    @staticmethod
    def manual_seed(_seed: int) -> None:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        ApiInferenceRunner(
            config,
            candidate_adapter_path=args.candidate_adapter,
            output_dir=args.output_dir,
        ).run()
        return 0
    except Exception as exc:                                        # noqa: BLE001
        reason = (
            exc.reason_code
            if isinstance(exc, RunnerContractError)
            else classify_failure(message=str(exc), returncode=1)
        )
        print(
            "RUNNER_ERROR: " + json.dumps(
                {"reason_code": reason, "detail": str(exc)}, ensure_ascii=False),
            file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
