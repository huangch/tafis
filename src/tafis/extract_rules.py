"""
tafis.extract_rules

Extract human-inspectable fuzzy rules from a trained ANFISRegressor checkpoint.

What "rule extraction" means here:
  - Antecedent parameters:
      Gaussian membership centers and sigmas for each rule and each feature.
  - Consequent parameters:
      TSK linear weights and bias for each rule.
  - Rule importance statistics computed on a dataset:
      - avg_firing: average normalized firing strength of each rule over the dataset
      - coverage: fraction of samples where rule firing > threshold
      - top_examples: indices of samples where the rule fires strongest (for debugging)

Outputs:
  - rules.json: structured representation, ideal for downstream LLM summarization
  - rules.csv : flattened table for analysis
  - rules.md  : a readable markdown report (still "mathy", linguify.py makes it more natural)

Important:
  - This module does not require the target values y. It operates on X only.
  - For interpretability, you should run it on training data or a representative dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .data import FeatureMetadata, prepare_inference_data
from .model import ANFISRegressor


@dataclass
class ExtractConfig:
    """
    Configuration for rule extraction and importance statistics.
    """
    firing_threshold: float = 0.1
    top_k_rules: int = 20
    top_examples_per_rule: int = 5
    batch_size: int = 8192
    device: str = "cuda"


def _device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict:
    """
    Load a checkpoint created by tafis.train.save_checkpoint.
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if "model_state_dict" not in ckpt:
        raise ValueError("Invalid checkpoint: missing model_state_dict.")
    return ckpt


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> ANFISRegressor:
    """
    Reconstruct ANFISRegressor from a saved checkpoint.
    """
    num_features = ckpt.get("num_features")
    num_rules = ckpt.get("num_rules")
    if num_features is None or num_rules is None:
        raise ValueError("Checkpoint missing num_features/num_rules.")
    model = ANFISRegressor(num_features=int(num_features), num_rules=int(num_rules)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def compute_rule_stats(
    model: ANFISRegressor,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute normalized firing strengths for all samples and basic rule statistics.

    Returns:
      firing_all: (N, R) normalized firing strengths
      avg_firing: (R,) mean firing per rule
      coverage : (R,) fraction of samples where firing > threshold (threshold applied later)
    """
    xb = torch.from_numpy(X).float()
    firings: List[np.ndarray] = []

    for i in range(0, xb.shape[0], batch_size):
        x_batch = xb[i : i + batch_size].to(device, non_blocking=True)
        _y, firing = model(x_batch)
        firings.append(firing.detach().cpu().numpy())

    firing_all = np.concatenate(firings, axis=0).astype(np.float32, copy=False)
    avg_firing = firing_all.mean(axis=0).astype(np.float32, copy=False)
    # coverage computed with threshold in the caller so you can change threshold without recomputing firing
    coverage = np.zeros_like(avg_firing)
    return firing_all, avg_firing, coverage


def extract_rules_from_model(
    model: ANFISRegressor,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Extract raw rule parameters from the model (no data-dependent importance yet).

    Returns a dict:
      {
        "num_rules": R,
        "num_features": D,
        "features": [...],
        "rules": [
          {
            "rule_id": i,
            "antecedent": { feature: { "mf": "gaussian", "center": c, "sigma": s } ... },
            "consequent": { "bias": b, "weights": { feature: w, ... } }
          }, ...
        ]
      }
    """
    R = model.num_rules
    D = model.num_features
    if len(feature_names) != D:
        raise ValueError("feature_names length must equal model.num_features.")

    centers = model.mf.centers.detach().cpu().numpy().astype(np.float32)
    sigmas = np.exp(model.mf.log_sigmas.detach().cpu().numpy().astype(np.float32))
    weights = model.consequent.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.consequent.bias.detach().cpu().numpy().astype(np.float32)

    rules = []
    for i in range(R):
        antecedent = {}
        for j, feat in enumerate(feature_names):
            antecedent[feat] = {
                "mf": "gaussian",
                "center": float(centers[i, j]),
                "sigma": float(sigmas[i, j]),
            }
        consequent = {
            "bias": float(bias[i]),
            "weights": {feat: float(weights[i, j]) for j, feat in enumerate(feature_names)},
        }
        rules.append(
            {
                "rule_id": int(i),
                "antecedent": antecedent,
                "consequent": consequent,
            }
        )

    return {
        "num_rules": int(R),
        "num_features": int(D),
        "features": list(feature_names),
        "rules": rules,
    }


def attach_importance(
    rules_obj: Dict[str, Any],
    firing_all: np.ndarray,
    avg_firing: np.ndarray,
    firing_threshold: float,
    top_examples_per_rule: int,
) -> Dict[str, Any]:
    """
    Attach data-driven importance metrics to each rule in rules_obj in-place.

    Adds per rule:
      - avg_firing
      - coverage
      - top_examples (indices and firing strengths)

    Returns the updated object.
    """
    R = rules_obj["num_rules"]
    if firing_all.shape[1] != R:
        raise ValueError("firing_all does not match number of rules.")

    coverage = (firing_all > firing_threshold).mean(axis=0).astype(np.float32)

    for i, rule in enumerate(rules_obj["rules"]):
        rule["importance"] = {
            "avg_firing": float(avg_firing[i]),
            "coverage": float(coverage[i]),
            "firing_threshold": float(firing_threshold),
        }

        # Top examples: indices where this rule fires strongest
        top_idx = np.argsort(-firing_all[:, i])[:top_examples_per_rule]
        top_vals = firing_all[top_idx, i]
        rule["top_examples"] = [
            {"row_index": int(idx), "firing": float(val)} for idx, val in zip(top_idx, top_vals)
        ]

    # Add a summary section
    rules_obj["importance_summary"] = {
        "avg_firing_mean": float(avg_firing.mean()),
        "avg_firing_max": float(avg_firing.max()),
        "coverage_mean": float(coverage.mean()),
        "coverage_max": float(coverage.max()),
        "firing_threshold": float(firing_threshold),
    }
    return rules_obj


def rules_to_dataframe(rules_obj: Dict[str, Any]) -> pd.DataFrame:
    """
    Flatten rules into a DataFrame (one row per rule).

    Columns include:
      - rule_id
      - avg_firing, coverage
      - bias
      - weights_* (per feature)
      - center_* and sigma_* (per feature)
    """
    features = rules_obj["features"]
    rows = []
    for rule in rules_obj["rules"]:
        rid = rule["rule_id"]
        imp = rule.get("importance", {})
        row = {
            "rule_id": rid,
            "avg_firing": imp.get("avg_firing", np.nan),
            "coverage": imp.get("coverage", np.nan),
            "bias": rule["consequent"]["bias"],
        }
        for feat in features:
            row[f"w__{feat}"] = rule["consequent"]["weights"][feat]
            row[f"c__{feat}"] = rule["antecedent"][feat]["center"]
            row[f"s__{feat}"] = rule["antecedent"][feat]["sigma"]
        rows.append(row)
    return pd.DataFrame(rows)


def rules_to_markdown(rules_obj: Dict[str, Any], top_k_rules: int) -> str:
    """
    Render a readable Markdown report of the top rules by avg_firing.

    This is still a "math-like" representation. For Low/Medium/High mapping,
    use tafis.linguify afterwards.
    """
    rules = list(rules_obj["rules"])
    rules_sorted = sorted(
        rules, key=lambda r: r.get("importance", {}).get("avg_firing", 0.0), reverse=True
    )
    rules_sorted = rules_sorted[: min(top_k_rules, len(rules_sorted))]
    features = rules_obj["features"]

    lines = []
    lines.append("# Extracted ANFIS Rules")
    lines.append("")
    lines.append(f"- Num rules: {rules_obj['num_rules']}")
    lines.append(f"- Num features: {rules_obj['num_features']}")
    lines.append("")

    for rule in rules_sorted:
        rid = rule["rule_id"]
        imp = rule.get("importance", {})
        lines.append(f"## Rule {rid}")
        lines.append(f"- avg_firing: {imp.get('avg_firing', float('nan')):.6g}")
        lines.append(f"- coverage  : {imp.get('coverage', float('nan')):.6g}")
        lines.append("")
        lines.append("**Antecedent (Gaussian MF params):**")
        for feat in features:
            a = rule["antecedent"][feat]
            lines.append(f"- {feat}: center={a['center']:.4f}, sigma={a['sigma']:.4f}")
        lines.append("")
        lines.append("**Consequent (TSK linear):**")
        lines.append(f"- bias={rule['consequent']['bias']:.6g}")
        # Show top-weight features by absolute coefficient
        w = rule["consequent"]["weights"]
        topw = sorted(w.items(), key=lambda kv: abs(kv[1]), reverse=True)[: min(10, len(w))]
        for feat, coef in topw:
            lines.append(f"- {feat}: {coef:+.6g}")
        lines.append("")
        lines.append("**Top examples (row indices):**")
        for ex in rule.get("top_examples", []):
            lines.append(f"- idx={ex['row_index']}, firing={ex['firing']:.6g}")
        lines.append("")
    return "\n".join(lines)


def extract_rules_pipeline(
    csv_path: str | Path,
    checkpoint_path: str | Path,
    meta_path: str | Path,
    out_dir: str | Path,
    cfg: ExtractConfig,
) -> Dict[str, Any]:
    """
    End-to-end rule extraction:
      - Load training metadata
      - Prepare X from the given csv_path aligned to training features
      - Load checkpoint and model
      - Compute firing strengths and importance metrics
      - Save rules.{json,csv,md}

    Returns:
      rules_obj
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata = FeatureMetadata.from_json(meta_path)
    X, _df = prepare_inference_data(csv_path=csv_path, metadata=metadata)

    dev = _device(cfg.device)
    ckpt = load_checkpoint(checkpoint_path, dev)
    model = build_model_from_checkpoint(ckpt, dev)

    firing_all, avg_firing, _coverage = compute_rule_stats(
        model=model, X=X, batch_size=cfg.batch_size, device=dev
    )

    rules_obj = extract_rules_from_model(model=model, feature_names=metadata.feature_cols)
    rules_obj = attach_importance(
        rules_obj=rules_obj,
        firing_all=firing_all,
        avg_firing=avg_firing,
        firing_threshold=cfg.firing_threshold,
        top_examples_per_rule=cfg.top_examples_per_rule,
    )

    # Save JSON
    (out / "rules.json").write_text(json.dumps(rules_obj, indent=2), encoding="utf-8")

    # Save CSV
    df_rules = rules_to_dataframe(rules_obj)
    df_rules.to_csv(out / "rules.csv", index=False)

    # Save Markdown
    md = rules_to_markdown(rules_obj, top_k_rules=cfg.top_k_rules)
    (out / "rules.md").write_text(md, encoding="utf-8")

    return rules_obj
