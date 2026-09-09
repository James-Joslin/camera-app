"""Reusable contracts and strategies for the person-detection workflows."""

from .modeling.assignment import ATSSAnchorAssigner, AnchorAssigner, AssignmentResult
from .core.contracts import DetectionBackend, EvaluationBackend, ModelOptimizationPipeline
from .data.sampling import (
    DEFAULT_HARD_CASE_POLICY,
    GreedyStratifiedSelector,
    HardCaseSamplingPolicy,
    RecordSelectionStrategy,
)

__all__ = [
    "ATSSAnchorAssigner",
    "AnchorAssigner",
    "AssignmentResult",
    "DEFAULT_HARD_CASE_POLICY",
    "DetectionBackend",
    "EvaluationBackend",
    "GreedyStratifiedSelector",
    "HardCaseSamplingPolicy",
    "ModelOptimizationPipeline",
    "RecordSelectionStrategy",
]

