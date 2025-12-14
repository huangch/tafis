
# TAFIS

TAFIS is a practical pipeline for learning an interpretable fuzzy regression model from a CSV:

- CSV → numeric feature matrix (robust parsing + missing value handling)
- Train a GPU-capable TSK-ANFIS regressor (PyTorch)
- Predict on new CSV files using the same saved preprocessing metadata
- Extract fuzzy rules + importance statistics
- Convert rules into human-friendly Low/Medium/High form
- (Optional) Use an LLM to rewrite rules into a concise explanation

This repository is intentionally modular so you can replace parts (e.g., your own LLM gateway).


## Project structure
tafis/
pyproject.toml
README.md
configs/
example.yaml
src/
tafis/
init.py
cli.py
data.py
model.py
train.py
extract_rules.py
linguify.py
llm_summarize.py
utils.py
## Installation

From the project root:

```bash
pip install -e .
This installs a tafis CLI entrypoint defined in pyproject.toml.


Key concepts



1) 
target_col
 is always user-defined


You always specify the regression target column with --target_col.


2) Features are user-defined OR auto-selected


You can either:

•	explicitly specify feature columns via --features or --features_file, or
•	let TAFIS auto-select all numeric-like columns excluding (id_cols + target_col)


Auto-selection includes robust conversion:

•	true/false/y/n/yes/no/on/off/1/0 → 0/1
•	numeric strings → float
•	non-numeric columns are dropped (with a log in metadata)



3) Inference is reproducible via 
features_used.json


When you train, TAFIS writes:

•	features_used.json


This file captures:

•	the exact feature columns used
•	id columns
•	target column
•	duplicate-column rename mapping
•	per-feature median imputation values


For prediction, you MUST pass this metadata file, preventing feature drift.


Train (regression)



Train with auto feature selection

tafis train \
  --csv data.csv \
  --target_col protein_specificity_score \
  --num_rules 32 \
  --device cuda \
  --out runs/exp1
Artifacts written to runs/exp1/:

•	features_used.json
•	feature_cols_used.txt
•	id_cols.txt
•	dropped_cols.json
•	rename_map.json
•	best.pt, last.pt
•	train_log.json
•	train_indices.npy, val_indices.npy



Train with explicit features


Put your features into a text file features.txt:
# one feature per line
known_TAA
Tau
SC_norm
RNA_target_prevalence
protein_safety_score
Then:
tafis train \
  --csv data.csv \
  --target_col protein_specificity_score \
  --features_file features.txt \
  --num_rules 32 \
  --device cuda \
  --out runs/exp1
Notes:

•	If a feature name is missing from the CSV, training fails fast.
•	If you accidentally include target_col or id_cols in features, training fails fast.



Predict on a new CSV

tafis predict \
  --csv new_data.csv \
  --checkpoint runs/exp1/best.pt \
  --meta runs/exp1/features_used.json \
  --out preds.csv \
  --device cuda
Include identifier columns in output:
tafis predict \
  --csv new_data.csv \
  --checkpoint runs/exp1/best.pt \
  --meta runs/exp1/features_used.json \
  --out preds.csv \
  --keep_cols UniProt_accession,GN
Optionally dump per-rule firing strengths (debug/explain):
tafis predict \
  --csv new_data.csv \
  --checkpoint runs/exp1/best.pt \
  --meta runs/exp1/features_used.json \
  --out preds.csv \
  --dump_firing

Extract fuzzy rules


Rule extraction requires:

•	a checkpoint (best.pt)
•	the training metadata (features_used.json)
•	a CSV to compute importance statistics (often the training CSV or representative data)


Currently, rule extraction is provided as a Python pipeline function:

•	tafis.extract_rules.extract_rules_pipeline(...)


If you want rule extraction on CLI, the next step is to wire it into tafis.cli
(using the existing pipeline function).

Example (Python):
from tafis.extract_rules import ExtractConfig, extract_rules_pipeline

rules = extract_rules_pipeline(
    csv_path="data.csv",
    checkpoint_path="runs/exp1/best.pt",
    meta_path="runs/exp1/features_used.json",
    out_dir="runs/exp1/rules",
    cfg=ExtractConfig(
        firing_threshold=0.1,
        top_k_rules=20,
        top_examples_per_rule=5,
        device="cuda",
    )
)
This creates:

•	runs/exp1/rules/rules.json
•	runs/exp1/rules/rules.csv
•	runs/exp1/rules/rules.md



Linguify (Low/Medium/High)


Convert numeric membership parameters into linguistic labels:

Example (Python):
from tafis.linguify import LinguifyConfig, linguify_files

ling = linguify_files(
    rules_json_path="runs/exp1/rules/rules.json",
    out_dir="runs/exp1/rules",
    cfg=LinguifyConfig(bins=(0.33, 0.66), sigma_bins=(0.12, 0.25), top_k_rules=20)
)
Outputs:

•	runs/exp1/rules/rules_linguified.json
•	runs/exp1/rules/rules_linguified.md



Optional: LLM summarization


tafis.llm_summarize provides:

•	strict prompt templates
•	a backend-agnostic interface


By default it does not call any API, because call_llm() is not implemented.
You should replace call_llm() with your own LLM integration.

Example (Python):
from tafis.llm_summarize import SummarizeConfig, summarize_rules_with_llm

md = summarize_rules_with_llm(
    rules_linguified_json_path="runs/exp1/rules/rules_linguified.json",
    out_path="runs/exp1/rules/summary.md",
    cfg=SummarizeConfig(top_k=15)
)

Notes on interpretability and stability


•	Increasing num_rules can improve fit but may reduce interpretability.
•	A small firing_sparsity_lambda can help rules become more distinct.
•	If your CSV contains duplicate column names, TAFIS renames them deterministically
(e.g., log2_Tumor_vs_Pan_normal__dup1). Always use the post-dedup names when
specifying --features.


License

MIT (as set in pyproject.toml).
