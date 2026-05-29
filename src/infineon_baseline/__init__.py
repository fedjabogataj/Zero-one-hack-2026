"""Statistical baseline + harness for the Industrial AI hackathon track."""
from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.1.0"

# Make generate_sequences.py (and validate_sequence) importable from training_data/
# without copying or vendoring its code.
_TRAINING_DATA = Path(__file__).resolve().parents[2] / "training_data"
if str(_TRAINING_DATA) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DATA))

from generate_sequences import validate_sequence, Violation  # noqa: E402

__all__ = ["__version__", "validate_sequence", "Violation"]
