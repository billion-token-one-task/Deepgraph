"""Offline datasets for fairness classification benchmarks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairlearn.datasets import fetch_bank_marketing, fetch_credit_card


@dataclass
class Dataset:
    name: str
    seed: int
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    sensitive_train: np.ndarray
    sensitive_test: np.ndarray


def _synthetic_grouped(seed: int, n_samples: int = 800) -> Dataset:
    rng = np.random.default_rng(seed)
    x, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=4,
        n_redundant=1,
        random_state=seed,
    )
    sensitive = (x[:, 0] + 0.8 * rng.normal(size=n_samples) > 0).astype(int)
    y = np.where((sensitive == 1) & (rng.random(n_samples) < 0.18), 1, y)
    y = np.where((sensitive == 0) & (rng.random(n_samples) < 0.08), 0, y)
    x = np.column_stack([x, sensitive])
    x_train, x_test, y_train, y_test, a_train, a_test = train_test_split(
        x,
        y,
        sensitive,
        test_size=0.35,
        random_state=seed + 11,
        stratify=y,
    )
    return Dataset(
        name="synthetic_grouped",
        seed=seed,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_train=a_train,
        sensitive_test=a_test,
    )


def _from_sklearn_table(name: str, seed: int, x: np.ndarray, y: np.ndarray,
                        sensitive_column: int, positive_classes: set[int]) -> Dataset:
    """Create a deterministic grouped binary dataset from bundled sklearn data."""
    x = np.asarray(x, dtype=float)
    y_raw = np.asarray(y)
    y_binary = np.array([1 if int(label) in positive_classes else 0 for label in y_raw], dtype=int)

    source = x[:, sensitive_column]
    threshold = float(np.median(source))
    sensitive = (source > threshold).astype(int)
    if len(np.unique(sensitive)) < 2:
        threshold = float(np.mean(source))
        sensitive = (source > threshold).astype(int)

    scaled = StandardScaler().fit_transform(x)
    features = np.column_stack([scaled, sensitive])
    x_train, x_test, y_train, y_test, a_train, a_test = train_test_split(
        features,
        y_binary,
        sensitive,
        test_size=0.35,
        random_state=seed + 17,
        stratify=y_binary,
    )
    return Dataset(
        name=name,
        seed=seed,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_train=a_train,
        sensitive_test=a_test,
    )


def _sklearn_breast_cancer_grouped(seed: int) -> Dataset:
    data = load_breast_cancer()
    return _from_sklearn_table(
        "sklearn_breast_cancer_grouped",
        seed,
        data.data,
        data.target,
        sensitive_column=0,
        positive_classes={1},
    )


def _sklearn_wine_grouped(seed: int) -> Dataset:
    data = load_wine()
    return _from_sklearn_table(
        "sklearn_wine_grouped",
        seed,
        data.data,
        data.target,
        sensitive_column=12,
        positive_classes={1, 2},
    )


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _from_openml_frame(
    name: str,
    seed: int,
    frame,
    target_column: str,
    sensitive_column: str,
    positive_target,
    sensitive_mapper,
) -> Dataset:
    """Convert an OpenML frame into the benchmark's binary grouped format."""
    if target_column not in frame or sensitive_column not in frame:
        raise ValueError(f"OpenML frame missing {target_column!r} or {sensitive_column!r}")

    y_raw = frame[target_column].astype(str).str.strip()
    y_binary = y_raw.map(positive_target).astype(int).to_numpy()
    sensitive = frame[sensitive_column].map(sensitive_mapper).astype(int).to_numpy()

    features = frame.drop(columns=[target_column, sensitive_column])
    numeric_columns = list(features.select_dtypes(include=["number", "bool"]).columns)
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(("num", StandardScaler(), numeric_columns))
    if categorical_columns:
        transformers.append(("cat", _one_hot_encoder(), categorical_columns))
    if not transformers:
        raise ValueError("OpenML frame has no usable feature columns")

    preprocessor = ColumnTransformer(transformers)
    encoded = preprocessor.fit_transform(features)
    x = np.column_stack([np.asarray(encoded, dtype=float), sensitive])
    x_train, x_test, y_train, y_test, a_train, a_test = train_test_split(
        x,
        y_binary,
        sensitive,
        test_size=0.35,
        random_state=seed + 23,
        stratify=y_binary,
    )
    return Dataset(
        name=name,
        seed=seed,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_train=a_train,
        sensitive_test=a_test,
    )


def _openml_adult_sex(seed: int) -> Dataset:
    data = fetch_openml(name="adult", version=2, as_frame=True)
    return _from_openml_frame(
        "openml_adult_sex",
        seed,
        data.frame,
        target_column="class",
        sensitive_column="sex",
        positive_target=lambda value: 1 if str(value).strip().startswith(">50K") else 0,
        sensitive_mapper=lambda value: 1 if str(value).strip().lower() == "male" else 0,
    )


def _openml_german_credit_sex(seed: int) -> Dataset:
    data = fetch_openml(name="credit-g", version=1, as_frame=True)
    return _from_openml_frame(
        "openml_german_credit_sex",
        seed,
        data.frame,
        target_column="class",
        sensitive_column="personal_status",
        positive_target=lambda value: 1 if str(value).strip().lower() == "good" else 0,
        sensitive_mapper=lambda value: 1 if str(value).strip().lower().startswith("male") else 0,
    )


def _fairlearn_credit_card_sex(seed: int) -> Dataset:
    data = fetch_credit_card(as_frame=True)
    return _from_openml_frame(
        "fairlearn_credit_card_sex",
        seed,
        data.frame,
        target_column="y",
        sensitive_column="x2",
        positive_target=lambda value: 1 if int(value) == 1 else 0,
        sensitive_mapper=lambda value: 1 if int(value) == 1 else 0,
    )


def _fairlearn_bank_marketing_age(seed: int) -> Dataset:
    data = fetch_bank_marketing(as_frame=True)
    frame = data.frame.copy()
    age_threshold = float(frame["V1"].astype(float).median())
    return _from_openml_frame(
        "fairlearn_bank_marketing_age",
        seed,
        frame,
        target_column="Class",
        sensitive_column="V1",
        positive_target=lambda value: 1 if str(value).strip().lower() in {"yes", "1", "true"} else 0,
        sensitive_mapper=lambda value: 1 if float(value) >= age_threshold else 0,
    )


def make_dataset(name: str, seed: int) -> Dataset:
    """Create a deterministic offline benchmark dataset."""
    if name == "synthetic_grouped":
        return _synthetic_grouped(seed)
    if name == "sklearn_breast_cancer_grouped":
        return _sklearn_breast_cancer_grouped(seed)
    if name == "sklearn_wine_grouped":
        return _sklearn_wine_grouped(seed)
    if name == "openml_adult_sex":
        return _openml_adult_sex(seed)
    if name == "openml_german_credit_sex":
        return _openml_german_credit_sex(seed)
    if name == "fairlearn_credit_card_sex":
        return _fairlearn_credit_card_sex(seed)
    if name == "fairlearn_bank_marketing_age":
        return _fairlearn_bank_marketing_age(seed)
    raise ValueError(f"unknown dataset: {name}")
