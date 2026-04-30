"""Methods for fairness classification benchmarks."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from benchmarks.fairness_classification.datasets import Dataset
from benchmarks.fairness_classification.metrics import compute_metrics


DEFAULT_FAIRNESS_PENALTY = 0.70
MAX_THRESHOLD_CANDIDATES = 128


class PreferenceConeThresholdClassifier:
    """Threshold search over group rates for a local fairness/accuracy proxy."""

    def __init__(self, fairness_penalty: float = DEFAULT_FAIRNESS_PENALTY):
        self.fairness_penalty = fairness_penalty
        self.estimator = LogisticRegression(max_iter=1000, solver="lbfgs")

    @staticmethod
    def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
        scores = np.unique(np.asarray(scores, dtype=float))
        if scores.size == 0:
            return np.array([0.5])
        if scores.size == 1:
            return np.array([0.0, scores[0], 1.0])
        if scores.size > MAX_THRESHOLD_CANDIDATES:
            quantiles = np.linspace(0.0, 1.0, MAX_THRESHOLD_CANDIDATES - 2)
            sampled = np.quantile(scores, quantiles)
            return np.unique(np.concatenate(([0.0], sampled, [1.0])))
        mids = (scores[:-1] + scores[1:]) / 2.0
        return np.unique(np.concatenate(([0.0], mids, [1.0])))

    def _best_global_threshold(self, probabilities: np.ndarray, y: np.ndarray) -> float:
        best_threshold = 0.5
        best_accuracy = -np.inf
        for threshold in self._candidate_thresholds(probabilities):
            predictions = probabilities >= threshold
            accuracy = float(np.mean(predictions == y))
            if accuracy > best_accuracy + 1e-12:
                best_accuracy = accuracy
                best_threshold = float(threshold)
        return best_threshold

    def fit(self, x: np.ndarray, y: np.ndarray, sensitive_features: np.ndarray):
        features = ordinary_predictive_features(x)
        self.estimator.fit(features, y)
        probabilities = self.estimator.predict_proba(features)[:, 1]
        groups = np.unique(sensitive_features)
        self.default_threshold_ = self._best_global_threshold(probabilities, y)
        self.thresholds_ = {group: self.default_threshold_ for group in groups}
        self.target_positive_rates_ = {}
        for group in groups:
            mask = sensitive_features == group
            self.target_positive_rates_[group] = (
                float(np.mean(probabilities[mask] >= self.default_threshold_)) if np.any(mask) else 0.0
            )

        if groups.size != 2:
            return self

        group_stats = {}
        for group in groups:
            mask = sensitive_features == group
            group_probabilities = probabilities[mask]
            group_labels = y[mask]
            stats = []
            for threshold in self._candidate_thresholds(group_probabilities):
                predictions = group_probabilities >= threshold
                correct = int(np.sum(predictions == group_labels))
                rate = float(np.mean(predictions)) if predictions.size else 0.0
                stats.append((float(threshold), correct, rate))
            group_stats[group] = stats

        n = len(y)
        g0, g1 = groups[0], groups[1]
        best_score = -np.inf
        best_accuracy = -np.inf
        best_gap = np.inf
        best_thresholds = (self.default_threshold_, self.default_threshold_)
        best_rates = (self.target_positive_rates_[g0], self.target_positive_rates_[g1])

        for threshold0, correct0, rate0 in group_stats[g0]:
            for threshold1, correct1, rate1 in group_stats[g1]:
                accuracy = (correct0 + correct1) / n
                gap = abs(rate1 - rate0)
                score = accuracy - self.fairness_penalty * gap
                if (
                    score > best_score + 1e-12
                    or (
                        abs(score - best_score) <= 1e-12
                        and (gap < best_gap - 1e-12 or (abs(gap - best_gap) <= 1e-12 and accuracy > best_accuracy))
                    )
                ):
                    best_score = score
                    best_accuracy = accuracy
                    best_gap = gap
                    best_thresholds = (threshold0, threshold1)
                    best_rates = (rate0, rate1)

        self.thresholds_ = {g0: best_thresholds[0], g1: best_thresholds[1]}
        self.target_positive_rates_ = {g0: best_rates[0], g1: best_rates[1]}
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.estimator.predict_proba(ordinary_predictive_features(x))[:, 1]
        sensitive_features = np.rint(x[:, -1]).astype(int)
        predictions = np.zeros(x.shape[0], dtype=int)
        for group in np.unique(sensitive_features):
            mask = sensitive_features == group
            group_indices = np.flatnonzero(mask)
            rate = getattr(self, "target_positive_rates_", {}).get(group)
            if rate is None:
                threshold = self.thresholds_.get(group, self.default_threshold_)
                predictions[mask] = (probabilities[mask] >= threshold).astype(int)
                continue

            k = int(np.rint(float(rate) * group_indices.size))
            k = max(0, min(group_indices.size, k))
            if k == 0:
                continue
            if k == group_indices.size:
                predictions[group_indices] = 1
                continue
            order = np.argsort(-probabilities[group_indices], kind="mergesort")
            predictions[group_indices[order[:k]]] = 1
        return predictions


def ordinary_predictive_features(x: np.ndarray) -> np.ndarray:
    """Return model features with the appended sensitive attribute removed."""
    if x.ndim != 2 or x.shape[1] < 2:
        return x
    return x[:, :-1]


def _logistic_regression(dataset: Dataset) -> np.ndarray:
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(ordinary_predictive_features(dataset.x_train), dataset.y_train)
    return model.predict(ordinary_predictive_features(dataset.x_test))


def _exponentiated_gradient(dataset: Dataset) -> np.ndarray:
    from fairlearn.reductions import DemographicParity, ExponentiatedGradient

    estimator = LogisticRegression(max_iter=1000, solver="lbfgs")
    model = ExponentiatedGradient(
        estimator,
        constraints=DemographicParity(),
        eps=0.02,
        max_iter=25,
    )
    model.fit(
        ordinary_predictive_features(dataset.x_train),
        dataset.y_train,
        sensitive_features=dataset.sensitive_train,
    )
    return model.predict(ordinary_predictive_features(dataset.x_test))


def _threshold_optimizer(dataset: Dataset) -> np.ndarray:
    from fairlearn.postprocessing import ThresholdOptimizer

    estimator = LogisticRegression(max_iter=1000, solver="lbfgs")
    estimator.fit(ordinary_predictive_features(dataset.x_train), dataset.y_train)
    model = ThresholdOptimizer(
        estimator=estimator,
        constraints="demographic_parity",
        prefit=True,
        predict_method="predict_proba",
    )
    model.fit(
        ordinary_predictive_features(dataset.x_train),
        dataset.y_train,
        sensitive_features=dataset.sensitive_train,
    )
    return model.predict(
        ordinary_predictive_features(dataset.x_test),
        sensitive_features=dataset.sensitive_test,
    )


def _preference_cone_threshold(dataset: Dataset) -> np.ndarray:
    model = PreferenceConeThresholdClassifier(fairness_penalty=DEFAULT_FAIRNESS_PENALTY)
    model.fit(dataset.x_train, dataset.y_train, dataset.sensitive_train)
    return model.predict(dataset.x_test)


def _preference_cone_threshold_with_penalty(dataset: Dataset, penalty: float) -> np.ndarray:
    model = PreferenceConeThresholdClassifier(fairness_penalty=penalty)
    model.fit(dataset.x_train, dataset.y_train, dataset.sensitive_train)
    return model.predict(dataset.x_test)


def _validation_selected_preference_cone(dataset: Dataset) -> np.ndarray:
    indices = np.arange(dataset.x_train.shape[0])
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=dataset.seed + 101,
        stratify=dataset.y_train,
    )
    candidates = []

    logistic = LogisticRegression(max_iter=1000, solver="lbfgs")
    logistic.fit(ordinary_predictive_features(dataset.x_train[train_idx]), dataset.y_train[train_idx])
    logistic_val = logistic.predict(ordinary_predictive_features(dataset.x_train[val_idx]))
    candidates.append((
        compute_metrics(dataset.y_train[val_idx], logistic_val, dataset.sensitive_train[val_idx])["fairness_score"],
        "logistic",
    ))

    for penalty in (0.20, 0.45, 0.70):
        model = PreferenceConeThresholdClassifier(fairness_penalty=penalty)
        model.fit(
            dataset.x_train[train_idx],
            dataset.y_train[train_idx],
            dataset.sensitive_train[train_idx],
        )
        val_pred = model.predict(dataset.x_train[val_idx])
        score = compute_metrics(
            dataset.y_train[val_idx],
            val_pred,
            dataset.sensitive_train[val_idx],
        )["fairness_score"]
        candidates.append((score, f"preference:{penalty}"))

    _, selected = max(candidates, key=lambda item: item[0])
    if selected == "logistic":
        final = LogisticRegression(max_iter=1000, solver="lbfgs")
        final.fit(ordinary_predictive_features(dataset.x_train), dataset.y_train)
        return final.predict(ordinary_predictive_features(dataset.x_test))

    penalty = float(selected.split(":", 1)[1])
    final = PreferenceConeThresholdClassifier(fairness_penalty=penalty)
    final.fit(dataset.x_train, dataset.y_train, dataset.sensitive_train)
    return final.predict(dataset.x_test)


def _validation_selected_fairlearn_baseline(dataset: Dataset) -> np.ndarray:
    indices = np.arange(dataset.x_train.shape[0])
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=dataset.seed + 211,
        stratify=dataset.y_train,
    )
    train_dataset = Dataset(
        name=dataset.name,
        seed=dataset.seed,
        x_train=dataset.x_train[train_idx],
        x_test=dataset.x_train[val_idx],
        y_train=dataset.y_train[train_idx],
        y_test=dataset.y_train[val_idx],
        sensitive_train=dataset.sensitive_train[train_idx],
        sensitive_test=dataset.sensitive_train[val_idx],
    )
    candidates = []
    for name, predictor in [
        ("logistic_regression", _logistic_regression),
        ("exponentiated_gradient", _exponentiated_gradient),
        ("threshold_optimizer", _threshold_optimizer),
    ]:
        try:
            val_pred = predictor(train_dataset)
            score = compute_metrics(
                train_dataset.y_test,
                val_pred,
                train_dataset.sensitive_test,
                fairness_penalty=DEFAULT_FAIRNESS_PENALTY,
            )["fairness_score"]
            candidates.append((score, name))
        except Exception:
            continue
    if not candidates:
        raise ValueError("no fairlearn baseline candidate could be validated")
    _, selected = max(candidates, key=lambda item: item[0])
    return {
        "logistic_regression": _logistic_regression,
        "exponentiated_gradient": _exponentiated_gradient,
        "threshold_optimizer": _threshold_optimizer,
    }[selected](dataset)


def fit_predict(method: str, dataset: Dataset) -> np.ndarray:
    """Fit a named method and return test predictions."""
    if method == "logistic_regression":
        return _logistic_regression(dataset)
    if method == "exponentiated_gradient":
        return _exponentiated_gradient(dataset)
    if method == "threshold_optimizer":
        return _threshold_optimizer(dataset)
    if method == "preference_cone_threshold":
        return _preference_cone_threshold(dataset)
    if method == "validation_selected_preference_cone":
        return _validation_selected_preference_cone(dataset)
    if method == "validation_selected_fairlearn_baseline":
        return _validation_selected_fairlearn_baseline(dataset)
    prefix = "preference_cone_threshold_penalty_"
    if method.startswith(prefix):
        try:
            penalty = float(method[len(prefix):])
        except ValueError as exc:
            raise ValueError(f"invalid preference cone penalty method: {method}") from exc
        return _preference_cone_threshold_with_penalty(dataset, penalty)
    raise ValueError(f"unknown method: {method}")
