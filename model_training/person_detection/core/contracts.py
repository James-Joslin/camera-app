"""Abstract contracts used at genuine backend and pipeline boundaries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DetectionBackend(ABC):
    """Inference backend contract shared by PyTorch and OpenVINO implementations."""

    @abstractmethod
    def predict(self, image: Any, score_threshold: float | None = None) -> list[dict]:
        """Return source-image XYXY detections for one image."""


class EvaluationBackend(ABC):
    """Metric backend contract for an accumulated detection result set."""

    @abstractmethod
    def evaluate(self, accumulator: Any, output_dir: Any) -> dict:
        """Evaluate accumulated predictions and persist backend artifacts."""


class ModelOptimizationPipeline(ABC):
    """A complete export/calibrate/validate/benchmark workflow."""

    @abstractmethod
    def run(self) -> dict:
        """Run optimization and return its machine-readable report."""

