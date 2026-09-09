"""Reusable training-weight and manifest-selection strategies."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_HARD_CASE_POLICY = {
    "name": "citypersons-hard-cases-v1",
    "replacement": True,
    "baseWeight": 1.0,
    "maximumWeight": 4.0,
    "multipliers": {
        "size:small": 2.0,
        "visibility:heavily_occluded": 2.5,
        "sourceLabel:rider": 1.75,
        "sourceLabel:sitting person": 2.0,
        "sourceLabel:person (other)": 2.25,
        "posture:unusual": 2.25,
    },
}


class RecordSelectionStrategy(ABC):
    """Select immutable manifest indices for a bounded dataset subset."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable algorithm name stored in provenance reports."""

    @abstractmethod
    def select(
        self, strata_by_index: Sequence[set[str]], sample_count: int, seed: int
    ) -> list[int]:
        """Return deterministic source indices."""


class GreedyStratifiedSelector(RecordSelectionStrategy):
    """Balance rare under-covered tags while retaining deterministic tie-breaking."""

    @property
    def name(self) -> str:
        return "deterministic-greedy-stratified-v1"

    def select(
        self, strata_by_index: Sequence[set[str]], sample_count: int, seed: int
    ) -> list[int]:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if sample_count >= len(strata_by_index):
            return list(range(len(strata_by_index)))

        rng = random.Random(seed)
        tie_order = list(range(len(strata_by_index)))
        rng.shuffle(tie_order)
        tie_rank = {index: rank for rank, index in enumerate(tie_order)}
        available = set(range(len(strata_by_index)))
        selected: list[int] = []
        selected_counts: Counter[str] = Counter()
        population_counts = Counter(
            stratum for strata in strata_by_index for stratum in set(strata)
        )

        while available and len(selected) < sample_count:
            def score(index: int):
                gain = sum(
                    1.0
                    / ((selected_counts[stratum] + 1) * population_counts[stratum])
                    for stratum in strata_by_index[index]
                )
                return gain, -tie_rank[index]

            chosen = max(available, key=score)
            available.remove(chosen)
            selected.append(chosen)
            selected_counts.update(strata_by_index[chosen])
        return selected


@dataclass(frozen=True)
class HardCaseSamplingPolicy:
    """Turn canonical hard-case tags into bounded training-only sample weights."""

    name: str
    base_weight: float
    maximum_weight: float
    multipliers: Mapping[str, float]
    replacement: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HardCaseSamplingPolicy":
        multipliers = value.get("multipliers")
        if not isinstance(multipliers, Mapping) or not multipliers:
            raise ValueError("Sampling policy must define non-empty multipliers")
        policy = cls(
            name=str(value.get("name", "custom-hard-case-policy")),
            base_weight=float(value.get("baseWeight", 1.0)),
            maximum_weight=float(value.get("maximumWeight", 4.0)),
            multipliers={key: float(weight) for key, weight in multipliers.items()},
            replacement=bool(value.get("replacement", True)),
        )
        if policy.base_weight <= 0 or policy.maximum_weight < policy.base_weight:
            raise ValueError("Sampling weights must be positive and correctly bounded")
        return policy

    def weight(self, strata: set[str]) -> float:
        matching = [
            multiplier
            for tag, multiplier in self.multipliers.items()
            if tag in strata
        ]
        return min(max([self.base_weight, *matching]), self.maximum_weight)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "replacement": self.replacement,
            "baseWeight": self.base_weight,
            "maximumWeight": self.maximum_weight,
            "multipliers": dict(self.multipliers),
        }
