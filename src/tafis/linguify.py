"""
tafis.linguify

Turn extracted numeric ANFIS rules into more human-readable, linguistic rules.

Input:
  - rules.json produced by tafis.extract_rules (Gaussian MF params + TSK consequents + importance)

Output:
  - rules_linguified.json
  - rules_linguified.md

Key idea:
  Your features are normalized to [0, 1]. Therefore we can map MF centers into linguistic bins:
    center < b1        => "Low"
    b1 <= center < b2  => "Medium"
    center >= b2       => "High"

We also optionally add a qualitative description for sigma:
  - narrow / medium / broad (based on sigma relative to expected [0,1] range)

Important:
  - This step does NOT change the model. It only changes representation.
  - The linguistic names are deterministic and user-adjustable (bins and sigma thresholds).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class LinguifyConfig:
    """
    Configuration for mapping numeric MF parameters into linguistic labels.
    """
    # Boundaries in [0,1] for Low/Medium/High
    bins: Tuple[float, float] = (0.33, 0.66)

    # Sigma (width) qualitative thresholds (in [0,1] feature space)
    # Typical Gaussian MF sigma for normalized data might be ~0.1-0.3.
    sigma_bins: Tuple[float, float] = (0.12, 0.25)  # narrow / medium / broad

    top_k_rules: int = 20  # for markdown report


def _level_from_center(center: float, bins: Tuple[float, float]) -> str:
    b1, b2 = bins
    if center < b1:
        return "Low"
    if center < b2:
        return "Medium"
    return "High"


def _spread_from_sigma(sigma: float, sigma_bins: Tuple[float, float]) -> str:
    s1, s2 = sigma_bins
    if sigma < s1:
        return "Narrow"
    if sigma < s2:
        return "Medium"
    return "Broad"


def linguify_rules_obj(rules_obj: Dict[str, Any], cfg: LinguifyConfig) -> Dict[str, Any]:
    """
    Convert numeric MF params into linguistic descriptions.

    Adds:
      rule["antecedent_linguistic"][feature] = {
          "level": "Low/Medium/High",
          "spread": "Narrow/Medium/Broad",
          "center": float,
          "sigma": float
      }

    Also adds a compact string form:
      rule["if_then"] = "IF (feat1 is High) AND (feat2 is Low) THEN y = ..."

    Returns:
      A new dict (copy) with added linguistic fields.
    """
    out = json.loads(json.dumps(rules_obj))  # deep copy via JSON roundtrip
    features = out.get("features", [])
    bins = cfg.bins
    sigma_bins = cfg.sigma_bins

    for rule in out.get("rules", []):
        ant = rule.get("antecedent", {})
        ant_ling = {}
        parts = []
        for feat in features:
            a = ant[feat]
            c = float(a["center"])
            s = float(a["sigma"])
            level = _level_from_center(c, bins=bins)
            spread = _spread_from_sigma(s, sigma_bins=sigma_bins)
            ant_ling[feat] = {
                "level": level,
                "spread": spread,
                "center": c,
                "sigma": s,
                "mf": a.get("mf", "gaussian"),
            }
            parts.append(f"({feat} is {level})")

        # Consequent string (TSK)
        cons = rule.get("consequent", {})
        bias = float(cons.get("bias", 0.0))
        weights = cons.get("weights", {})

        # Compact: show top weights by abs value
        items = sorted(weights.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
        items = items[: min(8, len(items))]
        terms = [f"{float(w):+.4g}*{feat}" for feat, w in items]
        cons_str = " + ".join(terms)
        cons_str = f"{cons_str} {bias:+.4g}".strip()

        rule["antecedent_linguistic"] = ant_ling
        rule["if_then"] = f"IF " + " AND ".join(parts) + f" THEN y = {cons_str}"

    out["linguify_config"] = {
        "bins": list(cfg.bins),
        "sigma_bins": list(cfg.sigma_bins),
    }
    return out


def linguified_to_markdown(rules_obj_ling: Dict[str, Any], cfg: LinguifyConfig) -> str:
    """
    Render a Markdown report emphasizing linguistic rules and importance.
    """
    rules = list(rules_obj_ling.get("rules", []))
    rules_sorted = sorted(
        rules, key=lambda r: r.get("importance", {}).get("avg_firing", 0.0), reverse=True
    )
    rules_sorted = rules_sorted[: min(cfg.top_k_rules, len(rules_sorted))]

    lines: List[str] = []
    lines.append("# Linguified ANFIS Rules")
    lines.append("")
    lines.append(f"- Num rules: {rules_obj_ling.get('num_rules')}")
    lines.append(f"- Num features: {rules_obj_ling.get('num_features')}")
    lines.append(f"- Bins (Low/Med/High): {cfg.bins}")
    lines.append(f"- Sigma bins (Narrow/Med/Broad): {cfg.sigma_bins}")
    lines.append("")

    for rule in rules_sorted:
        rid = rule["rule_id"]
        imp = rule.get("importance", {})
        lines.append(f"## Rule {rid}")
        lines.append(f"- avg_firing: {imp.get('avg_firing', float('nan')):.6g}")
        lines.append(f"- coverage  : {imp.get('coverage', float('nan')):.6g}")
        lines.append("")
        lines.append("**IF–THEN (compact):**")
        lines.append("")
        lines.append(f"- {rule.get('if_then', '')}")
        lines.append("")
        lines.append("**Antecedent details:**")
        ant = rule.get("antecedent_linguistic", {})
        # Show features in stable order if present
        feats = rules_obj_ling.get("features", [])
        for feat in feats:
            a = ant.get(feat, {})
            if a:
                lines.append(
                    f"- {feat}: {a['level']} ({a['spread']})  "
                    f"(center={a['center']:.3f}, sigma={a['sigma']:.3f})"
                )
        lines.append("")
        lines.append("**Top examples (row indices):**")
        for ex in rule.get("top_examples", []):
            lines.append(f"- idx={ex['row_index']}, firing={ex['firing']:.6g}")
        lines.append("")

    return "\n".join(lines)


def linguify_files(
    rules_json_path: str | Path,
    out_dir: str | Path,
    cfg: LinguifyConfig,
) -> Dict[str, Any]:
    """
    Load rules.json, linguify, and save outputs.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rules_obj = json.loads(Path(rules_json_path).read_text(encoding="utf-8"))
    rules_ling = linguify_rules_obj(rules_obj, cfg)

    (out / "rules_linguified.json").write_text(
        json.dumps(rules_ling, indent=2), encoding="utf-8"
    )
    md = linguified_to_markdown(rules_ling, cfg)
    (out / "rules_linguified.md").write_text(md, encoding="utf-8")

    return rules_ling
