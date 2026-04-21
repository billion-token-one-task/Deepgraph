"""Optional MLflow-backed tracking for DeepGraph runs and bundles."""

from __future__ import annotations

import contextlib

from config import MLFLOW_TRACKING_URI


def _mlflow():
    if not MLFLOW_TRACKING_URI:
        return None
    try:
        import mlflow
    except Exception:
        return None
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow


@contextlib.contextmanager
def tracked_run(name: str, tags: dict | None = None):
    mlflow = _mlflow()
    if mlflow is None:
        yield None
        return
    with mlflow.start_run(run_name=name):
        for key, value in (tags or {}).items():
            mlflow.set_tag(key, value)
        yield mlflow


def log_metrics(metrics: dict[str, float | int | None]) -> None:
    mlflow = _mlflow()
    if mlflow is None:
        return
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            mlflow.log_metric(key, float(value))
        except Exception:
            continue


def log_artifact(path: str) -> None:
    mlflow = _mlflow()
    if mlflow is None:
        return
    try:
        mlflow.log_artifact(path)
    except Exception:
        return
