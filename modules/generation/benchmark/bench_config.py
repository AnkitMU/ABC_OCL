"""
Benchmark configuration schema and defaults.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json

FAMILY_KEYS = [
    "cardinality", "uniqueness", "navigation", "quantified",
    "arithmetic", "string", "enum", "type_checks"
]

OPERATORS = [
    "forAll", "exists", "select", "collect", "size", "isUnique",
    "implies", "oclIsKindOf", "oclAsType", "oclIsUndefined", "oclIsInvalid"
]

TYPES = ["Integer", "Real", "String", "Boolean", "Enum"]


@dataclass
class RedundancyConfig:
    similarity_threshold: float = 0.85
    implication_mode: str = "off"  # off|greedy|budgeted
    implication_budget_ms: int = 0
    novelty_boost: bool = True


@dataclass
class EdgeCaseConfig:
    enable: bool = False
    boundary_min: int = 0
    empty_collections_min: int = 0
    deep_navigation_min: int = 0
    mixed_quantifiers_min: int = 0


@dataclass
class LibraryConfig:
    enabled: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Dict] = field(default_factory=dict)
    max_repeats_per_context: Dict[str, int] = field(default_factory=dict)


@dataclass
class CoverageTargets:
    class_context_pct: int = 80
    attribute_ref_pct: int = 60
    association_ref_pct: int = 70
    assoc_both_ends_pct: int = 50
    operator_mins: Dict[str, int] = field(default_factory=lambda: {op: 0 for op in OPERATORS})
    nav_hops: Dict[str, int] = field(default_factory=lambda: {"0": 0, "1": 0, "2plus": 0})
    quantifier_depth: Dict[str, int] = field(default_factory=lambda: {"0": 0, "1": 0, "2plus": 0})
    types_min: Dict[str, int] = field(default_factory=lambda: {t: 0 for t in TYPES})
    enum_literals_min: Dict[str, int] = field(default_factory=dict)
    difficulty_mix: Dict[str, int] = field(default_factory=lambda: {"easy": 40, "medium": 40, "hard": 20})


@dataclass
class QuantitiesConfig:
    invariants: int = 20
    pre: int = 0
    post: int = 0
    per_class_min: int = 1
    per_class_max: int = 5
    per_assoc_min: int = 0
    per_assoc_max: int = 3
    families_pct: Dict[str, int] = field(default_factory=lambda: {
        "cardinality": 25,
        "uniqueness": 20,
        "navigation": 15,
        "quantified": 15,
        "arithmetic": 10,
        "string": 10,
        "enum": 5,
        "type_checks": 0,
    })


@dataclass
class BenchmarkProfile:
    quantities: QuantitiesConfig = field(default_factory=QuantitiesConfig)
    coverage: CoverageTargets = field(default_factory=CoverageTargets)
    library: LibraryConfig = field(default_factory=LibraryConfig)
    redundancy: RedundancyConfig = field(default_factory=RedundancyConfig)
    edge_cases: EdgeCaseConfig = field(default_factory=EdgeCaseConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkProfile":
        """Build profile from a dictionary (e.g. from YAML/JSON)."""
        def merge(dc, src):
            for k, v in src.items():
                if isinstance(v, dict) and hasattr(dc, k):
                    merge(getattr(dc, k), v)
                else:
                    setattr(dc, k, v)
        prof = cls()
        merge(prof, data)
        return prof

    @classmethod
    def from_json(cls, path: str) -> "BenchmarkProfile":
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
