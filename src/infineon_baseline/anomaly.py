"""Anomaly detection strategies for Task 3."""
from __future__ import annotations

from dataclasses import dataclass

from generate_sequences import validate_sequence


@dataclass
class AnomalyResult:
    is_valid: int       # 1 or 0
    score: float        # in [0.0, 1.0]; P(valid)
    predicted_rule: str # rule id when invalid; "" when valid


def detect_oracle(steps: list[str]) -> AnomalyResult:
    """Use the rule oracle: any reported violation → invalid + first rule reported."""
    violations = validate_sequence(steps)
    if violations:
        return AnomalyResult(is_valid=0, score=0.0, predicted_rule=violations[0].rule)
    return AnomalyResult(is_valid=1, score=1.0, predicted_rule="")
