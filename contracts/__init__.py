"""Structured contracts for DeepGraph pipeline boundaries."""

from contracts.base import ContractValidationError, SCHEMA_VERSION
from contracts.pipeline import (
    DeepInsightSpec,
    DiscoverySignalBundle,
    ExperimentIterationPacket,
    ExperimentJudgement,
    ExperimentResultPacket,
    ExperimentSpec,
    ManuscriptInputState,
    StructuredPaperRecord,
    merge_signal_bundle_metadata,
    normalize_deep_insight_storage,
    parse_deep_insight_spec,
)

__all__ = [
    "ContractValidationError",
    "SCHEMA_VERSION",
    "DeepInsightSpec",
    "DiscoverySignalBundle",
    "ExperimentIterationPacket",
    "ExperimentJudgement",
    "ExperimentResultPacket",
    "ExperimentSpec",
    "ManuscriptInputState",
    "StructuredPaperRecord",
    "merge_signal_bundle_metadata",
    "normalize_deep_insight_storage",
    "parse_deep_insight_spec",
]
