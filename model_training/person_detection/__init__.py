"""Reusable contracts and strategies for the person-detection workflows."""

from .assignment import ATSSAnchorAssigner, AnchorAssigner, AssignmentResult
from .contracts import DetectionBackend, EvaluationBackend, ModelOptimizationPipeline
from .sampling import (
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

