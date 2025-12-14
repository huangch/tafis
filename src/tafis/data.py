from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


_TRUE_SET = {"true", "t", "1", "yes", "y", "on"}
_FALSE_SET = {"false", "f", "0", "no", "n", "off"}


def _coerce_boolish_to_float(x: Any) -> Any:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().lower()
    if s in _TRUE_SET:
        return 1.0
    if s in _FALSE_SET:
        return 0.0
    return x


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    # Convert boolish strings to 0/1 first, then numeric
    s2 = s.map(_coerce_boolish_to_float)
    return pd.to_numeric(s2, errors="coerce")


def _dedup_columns(cols: List[str]) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    """
    Deduplicate column names deterministically.

    Returns:
      new_cols: renamed columns
      rename_map: original_name -> new_name (only for renamed duplicates)
      dropped_cols: not dropped here, but returned for reporting compatibility
    """
    seen = {}
    new_cols = []
    rename_map = {}
    dup_groups: Dict[str, List[str]] = {}

    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_name = f"{c}__dup{seen[c]}"
            new_cols.append(new_name)
            rename_map[c] = new_name
            dup_groups.setdefault(c, []).append(new_name)

    return new_cols, rename_map, dup_groups


def _is_already_01(arr: np.ndarray, tol: float = 1e-6) -> bool:
    if arr.size == 0:
        return True
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return True
    mn = float(np.min(finite))
    mx = float(np.max(finite))
    return (mn >= -tol) and (mx <= 1.0 + tol)


def _minmax_fit(arr: np.ndarray) -> Tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.min(finite)), float(np.max(finite))


def _minmax_transform(arr: np.ndarray, mn: float, mx: float) -> np.ndarray:
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        # constant or invalid -> map to zeros
        out = np.zeros_like(arr, dtype=np.float32)
        out[~np.isfinite(arr)] = np.nan
        return out
    out = (arr - mn) / (mx - mn)
    out = out.astype(np.float32, copy=False)
    return out


@dataclass
class FeatureMetadata:
    """
    Captures training-time preprocessing so inference is identical.

    New fields added:
      - normalization: "minmax" or "none"
      - scaler_min / scaler_max: dict feature->float
      - normalized_features: features that were actually transformed (not already [0,1])
    """
    target_col: str
    id_cols: List[str]
    feature_cols: List[str]
    impute_median: Dict[str, float]

    normalization: str = "minmax"  # "minmax" or "none"
    scaler_min: Dict[str, float] = None
    scaler_max: Dict[str, float] = None
    normalized_features: List[str] = None

    rename_map: Dict[str, str] = None
    dropped_cols: Dict[str, Any] = None

    def to_json(self, path: str | Path) -> None:
        obj = {
            "target_col": self.target_col,
            "id_cols": self.id_cols,
            "feature_cols": self.feature_cols,
            "impute_median": self.impute_median,
            "normalization": self.normalization,
            "scaler_min": self.scaler_min or {},
            "scaler_max": self.scaler_max or {},
            "normalized_features": self.normalized_features or [],
            "rename_map": self.rename_map or {},
            "dropped_cols": self.dropped_cols or {},
        }
        Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: str | Path) -> "FeatureMetadata":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return FeatureMetadata(
            target_col=obj["target_col"],
            id_cols=list(obj.get("id_cols", [])),
            feature_cols=list(obj.get("feature_cols", [])),
            impute_median=dict(obj.get("impute_median", {})),
            normalization=obj.get("normalization", "none"),
            scaler_min=dict(obj.get("scaler_min", {})),
            scaler_max=dict(obj.get("scaler_max", {})),
            normalized_features=list(obj.get("normalized_features", [])),
            rename_map=dict(obj.get("rename_map", {})),
            dropped_cols=dict(obj.get("dropped_cols", {})),
        )


def _load_features_file(features_file: Optional[Path]) -> Optional[List[str]]:
    if features_file is None:
        return None
    lines = Path(features_file).read_text(encoding="utf-8").splitlines()
    feats = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        feats.append(s)
    return feats if feats else None


def _select_features(
    df: pd.DataFrame,
    target_col: str,
    id_cols: List[str],
    features: Optional[List[str]],
    features_file: Optional[Path],
) -> Tuple[List[str], Dict[str, Any]]:
    dropped: Dict[str, Any] = {"non_numeric": [], "missing_requested": []}

    explicit = features or _load_features_file(features_file)
    if explicit is not None:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            dropped["missing_requested"] = missing
            raise ValueError(f"Requested feature columns missing from CSV: {missing}")
        # also guard against leakage columns
        for c in explicit:
            if c == target_col:
                raise ValueError("target_col cannot be used as a feature.")
            if c in id_cols:
                raise ValueError(f"id_col '{c}' cannot be used as a feature.")
        return explicit, dropped

    # Auto-select: all columns except id_cols + target, keep numeric-like after coercion
    candidates = [c for c in df.columns if (c not in set(id_cols)) and (c != target_col)]
    numeric_like = []
    for c in candidates:
        s = _coerce_numeric_series(df[c])
        if s.notna().any():
            numeric_like.append(c)
        else:
            dropped["non_numeric"].append(c)
    return numeric_like, dropped


def _apply_minmax_normalization_train(
    X: np.ndarray,
    feature_cols: List[str],
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float], List[str]]:
    """
    Fit per-feature min/max on training data and normalize to [0,1] if needed.

    Returns:
      Xn, scaler_min, scaler_max, normalized_features
    """
    scaler_min: Dict[str, float] = {}
    scaler_max: Dict[str, float] = {}
    normalized_features: List[str] = []

    Xn = X.astype(np.float32, copy=True)

    for j, feat in enumerate(feature_cols):
        col = Xn[:, j].astype(np.float32, copy=False)
        mn, mx = _minmax_fit(col)
        scaler_min[feat] = mn
        scaler_max[feat] = mx

        # Decide whether to transform
        if _is_already_01(col, tol=tol):
            continue

        Xn[:, j] = _minmax_transform(col, mn, mx)
        normalized_features.append(feat)

    return Xn, scaler_min, scaler_max, normalized_features


def _apply_minmax_normalization_infer(
    X: np.ndarray,
    feature_cols: List[str],
    scaler_min: Dict[str, float],
    scaler_max: Dict[str, float],
) -> np.ndarray:
    """
    Apply training-fitted min/max scaling at inference.
    If a feature is missing scaler params, leave as-is (but this should not happen if metadata is valid).
    """
    Xn = X.astype(np.float32, copy=True)
    for j, feat in enumerate(feature_cols):
        mn = scaler_min.get(feat, None)
        mx = scaler_max.get(feat, None)
        if mn is None or mx is None:
            continue
        Xn[:, j] = _minmax_transform(Xn[:, j], float(mn), float(mx))
    return Xn


def prepare_training_data(
    csv_path: str | Path,
    target_col: str,
    id_cols: List[str],
    features: Optional[List[str]] = None,
    features_file: Optional[Path] = None,
    save_metadata_path: Optional[str | Path] = None,
    normalize_numeric: bool = True,
) -> Tuple[np.ndarray, np.ndarray, FeatureMetadata, pd.DataFrame]:
    """
    Load training CSV, auto-detect numeric features (if not explicitly provided),
    convert boolish columns, impute missing with medians, and optionally normalize numeric columns to [0,1].

    Returns:
      X: (N, D) float32
      y: (N,) float32
      meta: FeatureMetadata (includes impute medians and scaler params)
      df: the original dataframe (deduped columns) for optional downstream usage
    """
    df = pd.read_csv(csv_path)

    # Deduplicate columns deterministically
    new_cols, rename_map, dup_groups = _dedup_columns(list(df.columns))
    df.columns = new_cols

    # Select features
    feature_cols, dropped_cols = _select_features(
        df=df,
        target_col=target_col,
        id_cols=id_cols,
        features=features,
        features_file=features_file,
    )

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in CSV columns.")

    # Build X
    X_list = []
    impute_median: Dict[str, float] = {}

    for c in feature_cols:
        s = _coerce_numeric_series(df[c])
        arr = s.to_numpy(dtype=np.float32, copy=False)
        med = float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0
        impute_median[c] = med
        arr = np.where(np.isfinite(arr), arr, med).astype(np.float32, copy=False)
        X_list.append(arr)

    X = np.stack(X_list, axis=1) if X_list else np.zeros((len(df), 0), dtype=np.float32)

    # y
    y = _coerce_numeric_series(df[target_col]).to_numpy(dtype=np.float32, copy=False)
    if np.isnan(y).any():
        raise ValueError("Target column contains NaN after numeric coercion. Clean your data or drop rows.")

    # Normalize if needed (min-max to [0,1] using training stats)
    scaler_min: Dict[str, float] = {}
    scaler_max: Dict[str, float] = {}
    normalized_features: List[str] = []

    normalization = "none"
    if normalize_numeric and X.shape[1] > 0:
        X, scaler_min, scaler_max, normalized_features = _apply_minmax_normalization_train(
            X, feature_cols
        )
        normalization = "minmax"

    meta = FeatureMetadata(
        target_col=target_col,
        id_cols=list(id_cols),
        feature_cols=list(feature_cols),
        impute_median=impute_median,
        normalization=normalization,
        scaler_min=scaler_min,
        scaler_max=scaler_max,
        normalized_features=normalized_features,
        rename_map=rename_map,
        dropped_cols=dropped_cols,
    )

    if save_metadata_path is not None:
        Path(save_metadata_path).parent.mkdir(parents=True, exist_ok=True)
        meta.to_json(save_metadata_path)

        # Helpful sidecar lists
        (Path(save_metadata_path).parent / "feature_cols_used.txt").write_text(
            "\n".join(feature_cols) + "\n", encoding="utf-8"
        )
        (Path(save_metadata_path).parent / "id_cols.txt").write_text(
            "\n".join(id_cols) + "\n", encoding="utf-8"
        )
        (Path(save_metadata_path).parent / "dropped_cols.json").write_text(
            json.dumps(dropped_cols, indent=2), encoding="utf-8"
        )
        (Path(save_metadata_path).parent / "rename_map.json").write_text(
            json.dumps(rename_map, indent=2), encoding="utf-8"
        )

    return X.astype(np.float32, copy=False), y.astype(np.float32, copy=False), meta, df


def prepare_inference_data(
    csv_path: str | Path,
    metadata: FeatureMetadata,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Prepare inference X using training metadata:
      - same feature columns
      - same boolean parsing + numeric coercion
      - same median imputation (training medians)
      - same normalization scaling (training min/max) if enabled
    """
    df = pd.read_csv(csv_path)

    # Deduplicate input columns the same way (important if upstream CSV has duplicates)
    new_cols, _rename_map, _dup_groups = _dedup_columns(list(df.columns))
    df.columns = new_cols

    X_list = []
    for c in metadata.feature_cols:
        if c not in df.columns:
            raise ValueError(f"Inference CSV missing required feature column: '{c}'")
        s = _coerce_numeric_series(df[c])
        arr = s.to_numpy(dtype=np.float32, copy=False)

        med = float(metadata.impute_median.get(c, 0.0))
        arr = np.where(np.isfinite(arr), arr, med).astype(np.float32, copy=False)
        X_list.append(arr)

    X = np.stack(X_list, axis=1) if X_list else np.zeros((len(df), 0), dtype=np.float32)

    if metadata.normalization == "minmax" and X.shape[1] > 0:
        X = _apply_minmax_normalization_infer(X, metadata.feature_cols, metadata.scaler_min, metadata.scaler_max)

    return X.astype(np.float32, copy=False), df
