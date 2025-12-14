
"""
tafis.cli

Command Line Interface for the TAFIS project.

Commands:
  - train:       Train ANFIS regressor on a CSV
  - predict:     Predict on a new CSV using saved checkpoint + training metadata
  - extract-rules: Extract fuzzy rules + importance stats from a trained model
  - linguify:    Convert extracted rules into Low/Medium/High (human-readable)
  - llm-summarize: Optional LLM rewrite of linguified rules into a concise report
  - show-meta:   Print quick summary of features_used.json

Core constraints:
  - target_col is ALWAYS required for training (user decides).
  - feature columns are user-defined IF provided; otherwise auto-selected by tafis.data.
  - inference (predict) MUST reuse training-time metadata (features_used.json).

Usage examples:

  # Train (auto feature selection)
  tafis train --csv data.csv --target_col protein_specificity_score --num_rules 32 --out runs/exp1

  # Predict
  tafis predict --csv new_data.csv --checkpoint runs/exp1/best.pt --meta runs/exp1/features_used.json --out preds.csv

  # Extract rules
  tafis extract-rules --csv data.csv --checkpoint runs/exp1/best.pt --meta runs/exp1/features_used.json --out runs/exp1/rules

  # Linguify rules (Low/Medium/High)
  tafis linguify --rules_json runs/exp1/rules/rules.json --out runs/exp1/rules

  # LLM summarize (requires you to implement tafis.llm_summarize.call_llm)
  tafis llm-summarize --rules_linguified_json runs/exp1/rules/rules_linguified.json --out runs/exp1/rules/summary.md
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import typer
from rich.console import Console
from rich.table import Table

from .data import FeatureMetadata, prepare_inference_data, prepare_training_data
from .extract_rules import ExtractConfig, extract_rules_pipeline
from .linguify import LinguifyConfig, linguify_files
from .llm_summarize import SummarizeConfig, summarize_rules_with_llm
from .model import ANFISRegressor
from .train import TrainConfig, train_anfis_regressor

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts if parts else None


def _load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if "model_state_dict" not in ckpt:
        raise ValueError("Invalid checkpoint: missing model_state_dict.")
    return ckpt


@app.command("train")
def cmd_train(
    csv: Path = typer.Option(..., "--csv", help="Input CSV path."),
    target_col: str = typer.Option(..., "--target_col", help="Regression target column in the CSV."),
    out: Path = typer.Option(..., "--out", help="Output directory for run artifacts."),
    # Feature selection
    features: Optional[str] = typer.Option(None, "--features", help="Comma-separated feature column names."),
    features_file: Optional[Path] = typer.Option(None, "--features_file", help="Text file: one feature column per line."),
    id_cols: str = typer.Option(
        "UniProt_accession,UniProt_proteinID,GN",
        "--id_cols",
        help="Comma-separated id columns to exclude from features.",
    ),
    # ANFIS / training hyperparams
    num_rules: int = typer.Option(32, "--num_rules", help="Number of fuzzy rules (user-controlled)."),
    epochs: int = typer.Option(200, "--epochs", help="Training epochs."),
    batch_size: int = typer.Option(4096, "--batch_size", help="Batch size."),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate."),
    weight_decay: float = typer.Option(1e-5, "--weight_decay", help="Weight decay (L2 regularization)."),
    val_size: float = typer.Option(0.1, "--val_size", help="Validation split fraction."),
    firing_sparsity_lambda: float = typer.Option(
        0.0,
        "--firing_sparsity_lambda",
        help="Optional penalty to encourage sparse rule usage (improves interpretability).",
    ),
    seed: int = typer.Option(1337, "--seed", help="Random seed."),
    device: str = typer.Option("cuda", "--device", help="cuda or cpu"),
):
    id_cols_list = _parse_csv_list(id_cols) or []
    feature_list = _parse_csv_list(features)

    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "features_used.json"

    console.print("[bold]Loading and preparing training data...[/bold]")
    X, y, meta, _df = prepare_training_data(
        csv_path=csv,
        target_col=target_col,
        id_cols=id_cols_list,
        features=feature_list,
        features_file=features_file,
        save_metadata_path=meta_path,
    )

    console.print(f"Prepared X shape: {X.shape}, y shape: {y.shape}")
    console.print(f"Using {len(meta.feature_cols)} feature columns.")

    cfg = TrainConfig(
        num_rules=num_rules,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        firing_sparsity_lambda=firing_sparsity_lambda,
        val_size=val_size,
        seed=seed,
        device=device,
    )

    console.print("[bold]Training ANFIS regressor...[/bold]")
    summary = train_anfis_regressor(X=X, y=y, meta=meta, out_dir=out, cfg=cfg)

    table = Table(title="Training Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in summary.items():
        table.add_row(k, f"{v:.6g}" if isinstance(v, float) else str(v))
    console.print(table)

    console.print(f"[green]Done.[/green] Artifacts saved to: {out}")


@app.command("predict")
def cmd_predict(
    csv: Path = typer.Option(..., "--csv", help="New CSV path for inference."),
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to best.pt (or last.pt)."),
    meta: Path = typer.Option(..., "--meta", help="Path to features_used.json from training."),
    out: Path = typer.Option(..., "--out", help="Output CSV path for predictions."),
    keep_cols: Optional[str] = typer.Option(
        None,
        "--keep_cols",
        help="Comma-separated columns to include in output (e.g., UniProt_accession,GN).",
    ),
    dump_firing: bool = typer.Option(
        False,
        "--dump_firing",
        help="If set, also output per-rule firing strengths (can make output wide).",
    ),
    device: str = typer.Option("cuda", "--device", help="cuda or cpu"),
):
    keep_cols_list = _parse_csv_list(keep_cols) or []

    metadata = FeatureMetadata.from_json(meta)
    console.print("[bold]Preparing inference data (aligned to training metadata)...[/bold]")
    X, df = prepare_inference_data(csv_path=csv, metadata=metadata)

    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    ckpt = _load_checkpoint(checkpoint, dev)

    num_features = ckpt.get("num_features")
    num_rules = ckpt.get("num_rules")
    if num_features is None or num_rules is None:
        raise ValueError("Checkpoint missing num_features/num_rules; cannot reconstruct model.")

    model = ANFISRegressor(num_features=int(num_features), num_rules=int(num_rules)).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    xb = torch.from_numpy(X).float()
    batch_size = 8192
    preds = []
    firings = []

    with torch.no_grad():
        for i in range(0, xb.shape[0], batch_size):
            x_batch = xb[i : i + batch_size].to(dev, non_blocking=True)
            y_pred, firing = model(x_batch)
            preds.append(y_pred.detach().cpu().numpy())
            if dump_firing:
                firings.append(firing.detach().cpu().numpy())

    pred = np.concatenate(preds, axis=0)

    out_df = pd.DataFrame({"prediction": pred.astype(np.float32)})

    for c in keep_cols_list:
        if c in df.columns:
            out_df[c] = df[c].values
        else:
            console.print(f"[yellow]Warning:[/yellow] keep_col '{c}' not found in input CSV.")

    if dump_firing:
        firing_all = np.concatenate(firings, axis=0)
        for r in range(firing_all.shape[1]):
            out_df[f"firing_rule_{r}"] = firing_all[:, r].astype(np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    console.print(f"[green]Done.[/green] Wrote predictions to: {out}")


@app.command("extract-rules")
def cmd_extract_rules(
    csv: Path = typer.Option(..., "--csv", help="CSV used to compute rule importance (train or representative)."),
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to best.pt (or last.pt)."),
    meta: Path = typer.Option(..., "--meta", help="Path to features_used.json from training."),
    out: Path = typer.Option(..., "--out", help="Output directory for extracted rules."),
    firing_threshold: float = typer.Option(0.1, "--firing_threshold", help="Coverage threshold on normalized firing."),
    top_k_rules: int = typer.Option(20, "--top_k_rules", help="Top K rules to render in Markdown."),
    top_examples_per_rule: int = typer.Option(5, "--top_examples_per_rule", help="Top example rows per rule."),
    batch_size: int = typer.Option(8192, "--batch_size", help="Batch size for firing computation."),
    device: str = typer.Option("cuda", "--device", help="cuda or cpu"),
):
    """
    Extract fuzzy rules and compute importance statistics.

    Writes:
      - rules.json
      - rules.csv
      - rules.md
    """
    cfg = ExtractConfig(
        firing_threshold=firing_threshold,
        top_k_rules=top_k_rules,
        top_examples_per_rule=top_examples_per_rule,
        batch_size=batch_size,
        device=device,
    )

    console.print("[bold]Extracting rules...[/bold]")
    extract_rules_pipeline(
        csv_path=csv,
        checkpoint_path=checkpoint,
        meta_path=meta,
        out_dir=out,
        cfg=cfg,
    )
    console.print(f"[green]Done.[/green] Rules saved to: {out}")


@app.command("linguify")
def cmd_linguify(
    rules_json: Path = typer.Option(..., "--rules_json", help="Path to rules.json produced by extract-rules."),
    out: Path = typer.Option(..., "--out", help="Output directory for linguified rules."),
    bins: str = typer.Option("0.33,0.66", "--bins", help="Two floats: Low/Med/High boundaries in [0,1]."),
    sigma_bins: str = typer.Option("0.12,0.25", "--sigma_bins", help="Two floats: Narrow/Med/Broad sigma boundaries."),
    top_k_rules: int = typer.Option(20, "--top_k_rules", help="Top K rules to render in Markdown."),
):
    """
    Convert numeric MF parameters into Low/Medium/High (and sigma to Narrow/Med/Broad).

    Writes:
      - rules_linguified.json
      - rules_linguified.md
    """
    b = _parse_csv_list(bins)
    sb = _parse_csv_list(sigma_bins)
    if not b or len(b) != 2:
        raise ValueError("--bins must be two comma-separated floats, e.g. 0.33,0.66")
    if not sb or len(sb) != 2:
        raise ValueError("--sigma_bins must be two comma-separated floats, e.g. 0.12,0.25")

    cfg = LinguifyConfig(
        bins=(float(b[0]), float(b[1])),
        sigma_bins=(float(sb[0]), float(sb[1])),
        top_k_rules=top_k_rules,
    )

    console.print("[bold]Linguifying rules...[/bold]")
    linguify_files(
        rules_json_path=rules_json,
        out_dir=out,
        cfg=cfg,
    )
    console.print(f"[green]Done.[/green] Linguified rules saved to: {out}")


@app.command("llm-summarize")
def cmd_llm_summarize(
    rules_linguified_json: Path = typer.Option(..., "--rules_linguified_json", help="Path to rules_linguified.json."),
    out: Path = typer.Option(..., "--out", help="Output Markdown file, e.g. summary.md"),
    top_k: int = typer.Option(15, "--top_k", help="How many top rules to include in the prompt."),
):
    """
    Optional: call an LLM to rewrite linguified rules into a concise report.

    NOTE:
      By default, tafis.llm_summarize.call_llm() is NotImplemented.
      You must implement call_llm() for your environment (OpenAI/Azure/internal gateway).
    """
    cfg = SummarizeConfig(top_k=top_k)

    console.print("[bold]Summarizing with LLM...[/bold]")
    try:
        summarize_rules_with_llm(
            rules_linguified_json_path=rules_linguified_json,
            out_path=out,
            cfg=cfg,
        )
    except NotImplementedError as e:
        console.print("[red]LLM call is not configured.[/red]")
        console.print(str(e))
        console.print(
            "Implement tafis.llm_summarize.call_llm(system_prompt, user_prompt, cfg) "
            "to enable this command, or set backend='none' in SummarizeConfig for a "
            "deterministic summary without an LLM."
        )
        raise typer.Exit(code=2)

    console.print(f"[green]Done.[/green] Wrote LLM summary to: {out}")


@app.command("show-meta")
def cmd_show_meta(
    meta: Path = typer.Option(..., "--meta", help="Path to features_used.json"),
):
    m = FeatureMetadata.from_json(meta)
    table = Table(title="Feature Metadata Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("target_col", m.target_col)
    table.add_row("n_features", str(len(m.feature_cols)))
    table.add_row("id_cols", ", ".join(m.id_cols))
    console.print(table)


if __name__ == "__main__":
    app()
