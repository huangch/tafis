"""
TAFIS: TSK-ANFIS regression pipeline.

This package provides:
  - CSV -> numeric features (with robust boolean parsing + auto feature selection)
  - GPU-capable ANFIS/TSK regression model training
  - Inference on new CSVs using saved training metadata
  - Rule extraction and linguistic rule rendering
  - Optional LLM-based rewriting of rules into human-friendly summaries

Public API policy:
  - We keep the public API minimal. Most users should use the CLI.
  - Internals can evolve; the CLI is the stable interface.
"""

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
