from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_SYSTEM_PROMPT = """You are a technical writer summarizing fuzzy inference rules extracted from a trained model.
STRICT RULES:
- Do NOT introduce new features, variables, or rules not present in the input JSON.
- Do NOT make causal claims. Only describe associations encoded in the rules.
- If something is ambiguous, say so.
- Keep the output concise, structured, and in Markdown.
"""


DEFAULT_USER_PROMPT_TEMPLATE = """You are given fuzzy rules extracted from a trained ANFIS/TSK model.
The rules are already mapped into linguistic labels (Low/Medium/High).

INPUT JSON:
    ```json
    {rules_json}

TASKS:
    1.	List the Top-{top_k} rules. For each rule:
        o	Start with "Rule <rule_id>:"
        o	Write ONE sentence describing the IF conditions and the general direction of the THEN part.
        o	Include "(avg_firing=…, coverage=…)" exactly once.
    2.	Cluster similar rules into 3-7 "Core Principles".
    3.	Add "How to Read" with 3-5 bullets explaining avg_firing, coverage, and TSK consequents.

OUTPUT FORMAT (Markdown):
    # Top Rules

    # Core Principles

    # How to Read

"""


@dataclass
class SummarizeConfig:
    """Configuration for LLM-based rule summarization.

    backend:
        - "llm"  (default) uses call_llm (NotImplemented by default).
        - "none" returns a deterministic Markdown summary without calling an LLM.
    """

    top_k: int = 15
    backend: str = "llm"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE


def _trim_top_k(rules_obj: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    """Return a shallow copy of rules_obj containing only the top_k rules by avg_firing."""

    rules = list(rules_obj.get("rules", []))
    rules_sorted = sorted(
        rules,
        key=lambda r: r.get("importance", {}).get("avg_firing", 0.0),
        reverse=True,
    )
    rules_sorted = rules_sorted[: min(top_k, len(rules_sorted))]
    return {
        "num_rules": rules_obj.get("num_rules"),
        "num_features": rules_obj.get("num_features"),
        "features": rules_obj.get("features"),
        "linguify_config": rules_obj.get("linguify_config"),
        "importance_summary": rules_obj.get("importance_summary"),
        "rules": rules_sorted,
    }


def _deterministic_summary(rules_obj: Dict[str, Any], top_k: int) -> str:
    """Produce a deterministic Markdown summary without calling any LLM."""

    rules = list(rules_obj.get("rules", []))
    rules_sorted = sorted(
        rules,
        key=lambda r: r.get("importance", {}).get("avg_firing", 0.0),
        reverse=True,
    )[: min(top_k, len(rules))]

    lines: List[str] = []
    lines.append("# Rule Summary (Deterministic)")
    lines.append("")
    lines.append("## Top Rules")
    lines.append("")

    for r in rules_sorted:
        rid = r.get("rule_id")
        imp = r.get("importance", {})
        avg_f = imp.get("avg_firing", float("nan"))
        cov = imp.get("coverage", float("nan"))
        if_then = r.get("if_then", "IF <conditions> THEN <TSK linear>")
        lines.append(f"- Rule {rid}: {if_then} (avg_firing={avg_f:.4g}, coverage={cov:.4g})")

    lines.append("")
    lines.append("## How to Read")
    lines.append("")
    lines.append("- avg_firing: average activation strength of a rule across samples.")
    lines.append("- coverage: fraction of samples where the rule activates above threshold.")
    lines.append("- IF part uses Low/Medium/High fuzzy membership labels.")
    lines.append(
        "- THEN part is a linear TSK consequent; coefficients indicate association only."
    )
    lines.append("")

    return "\n".join(lines)


def call_llm(system_prompt: str, user_prompt: str, cfg: SummarizeConfig) -> str:
    """Backend hook for calling an LLM.

    By default this function is not implemented. Replace it with your own
    integration (OpenAI, Azure, internal gateway, etc.).
    """

    raise NotImplementedError(
        "tafis.llm_summarize.call_llm is not implemented. "
        "Replace it with your own LLM integration."
    )


def summarize_rules(rules_linguified: Dict[str, Any], cfg: SummarizeConfig) -> str:
    """Summarize linguified rules into Markdown.

    - If cfg.backend == "none", returns a deterministic summary.
    - Otherwise, calls call_llm (which is a stub by default).
    """

    payload = _trim_top_k(rules_linguified, cfg.top_k)
    rules_json = json.dumps(payload, indent=2)

    if cfg.backend == "none":
        return _deterministic_summary(payload, cfg.top_k)

    user_prompt = cfg.user_prompt_template.format(
        rules_json=rules_json,
        top_k=cfg.top_k,
    )

    return call_llm(
        system_prompt=cfg.system_prompt,
        user_prompt=user_prompt,
        cfg=cfg,
    )


def summarize_rules_with_llm(
    rules_linguified_json_path: str | Path,
    out_path: str | Path,
    cfg: SummarizeConfig,
) -> str:
    """High-level helper used by the CLI.

    Loads rules_linguified_json_path, summarizes according to cfg, and writes
    Markdown to out_path. Returns the Markdown string.
    """

    rules_obj = json.loads(Path(rules_linguified_json_path).read_text(encoding="utf-8"))
    md = summarize_rules(rules_obj, cfg)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", encoding="utf-8")
    return md

