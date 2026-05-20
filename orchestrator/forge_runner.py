"""Async experiment forge lane so Auto Research scheduling is not blocked."""

from __future__ import annotations

import threading

from agents.experiment_forge import forge_experiment
from db import database as db
from orchestrator.pipeline import log_event

MAX_PARALLEL_FORGES = 2

_active_lock = threading.Lock()
_active_insight_ids: set[int] = set()


def active_forge_count() -> int:
    with _active_lock:
        return len(_active_insight_ids)


def is_forge_active(insight_id: int) -> bool:
    with _active_lock:
        return int(insight_id) in _active_insight_ids


def has_forge_capacity() -> bool:
    return active_forge_count() < MAX_PARALLEL_FORGES


def _mark_active(insight_id: int) -> None:
    with _active_lock:
        _active_insight_ids.add(int(insight_id))


def _mark_inactive(insight_id: int) -> None:
    with _active_lock:
        _active_insight_ids.discard(int(insight_id))


def _run_forge(insight_id: int, resource_class: str) -> None:
    _mark_active(insight_id)
    try:
        forged = forge_experiment(insight_id)
        from orchestrator.auto_research import handle_forge_completed

        handle_forge_completed(insight_id, forged, resource_class)
    except Exception as exc:
        from orchestrator.auto_research import handle_forge_failed

        handle_forge_failed(insight_id, str(exc))
        log_event("error", {"step": "forge_runner_failed", "insight_id": insight_id, "error": str(exc)})
    finally:
        _mark_inactive(insight_id)


def start_forge_async(*, insight_id: int, resource_class: str) -> bool:
    """Dispatch forge on a background thread; returns False if at capacity or already active."""
    if is_forge_active(insight_id) or not has_forge_capacity():
        return False
    thread = threading.Thread(
        target=_run_forge,
        args=(insight_id, resource_class),
        daemon=True,
        name=f"forge-insight-{insight_id}",
    )
    thread.start()
    log_event("forge_runner", {"step": "forge_dispatched", "insight_id": insight_id, "resource_class": resource_class})
    return True
