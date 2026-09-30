"""quickstart_xgboost: Task utilities (CSV-based)."""

import numpy as np
import xgboost as xgb
from pathlib import Path
import pandas as pd


# -----------------------------
# CSV LOADER
# -----------------------------
def load_csv_folder(path: Path):
    """
    Legge tutti i CSV in data/{client_id}/ e li concatena.
    """

    csv_files = sorted(path.glob("*.csv"))

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {path}")

    df = pd.concat(
        (pd.read_csv(f) for f in csv_files),
        ignore_index=True
    )

    return df


# -----------------------------
# DATAFRAME → DMATRIX
# -----------------------------
def dataframe_to_dmatrix(df: pd.DataFrame, label_col: str = "label"):
    """
    Converte DataFrame in XGBoost DMatrix.
    """

    y = df[label_col].to_numpy(dtype=np.float32)
    X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)

    return xgb.DMatrix(X, label=y)


# -----------------------------
# LOAD DATA (CLIENT SIDE)
# -----------------------------
def load_data(data_path: str | Path):
    """
    Carica dati da:
        data/{client_id}/*.csv

    e ritorna train/valid DMatrix.
    """

    data_path = Path(data_path)

    df = load_csv_folder(data_path)

    # shuffle deterministico
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # split
    split = int(len(df) * 0.8)

    train_df = df.iloc[:split]
    valid_df = df.iloc[split:]

    return (
        dataframe_to_dmatrix(train_df),
        dataframe_to_dmatrix(valid_df),
        len(train_df),
        len(valid_df),
    )


# -----------------------------
# CONFIG UTILS (FLOWER)
# -----------------------------
def replace_keys(input_dict, match="-", target="_"):
    """
    Ricorsivamente sostituisce "-" → "_" nelle chiavi.
    Utile per config Flower → XGBoost params.
    """

    out = {}

    for k, v in input_dict.items():
        new_k = k.replace(match, target)

        if isinstance(v, dict):
            out[new_k] = replace_keys(v, match, target)
        else:
            out[new_k] = v

    return out



from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def dataframe_to_dmatrix(df: pd.DataFrame) -> xgb.DMatrix:
    """Convert a labeled dataframe into an xgboost DMatrix."""
    y = df["label"].to_numpy(dtype=np.float32)
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    return xgb.DMatrix(X, label=y)


# def load_client_data(data_path: Path, train_frac: float = 0.8, seed: int = 42):
#     """Load all CSVs in a client folder and split into train/valid DMatrices."""
#     csv_files = sorted(Path(data_path).glob("*.csv"))

#     if not csv_files:
#         raise ValueError(f"No CSV files found in {data_path}")

#     df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
#     df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

#     split = int(len(df) * train_frac)
#     train_df = df.iloc[:split]
#     valid_df = df.iloc[split:]

#     return (
#         dataframe_to_dmatrix(train_df),
#         dataframe_to_dmatrix(valid_df),
#         len(train_df),
#         len(valid_df),
#     )


def build_xgb_params(cfg: dict) -> dict:
    """Build the xgboost params dict from the (unflattened) run config."""
    return {
        "objective": "reg:squarederror",
        "eta": cfg["params"]["eta"],
        "max_depth": cfg["params"]["max-depth"],
        "eval_metric": "rmse",
        "tree_method": "hist",
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE and RMSE between true and predicted values."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


import hashlib
import json


def _config_hash(
    train_frac: float,
    holdout_frac: float,
    seed: int,
    exclude_prefixes: list[str] | None,
    keep_only_prefixes: list[str] | None,
) -> str:
    """Fingerprint della configurazione che determina lo split."""
    payload = {
        "train_frac": train_frac,
        "holdout_frac": holdout_frac,
        "seed": seed,
        "exclude_prefixes": sorted(exclude_prefixes) if exclude_prefixes else None,
        "keep_only_prefixes": sorted(keep_only_prefixes) if keep_only_prefixes else None,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def load_client_data(
    data_path: Path,
    train_frac: float = 0.8,
    holdout_frac: float = 0.1,
    seed: int = 42,
    exclude_prefixes: list[str] | None = None,
    keep_only_prefixes: list[str] | None = None,
):
    data_path = Path(data_path)
    split_path = data_path / "split_assignment.csv"
    meta_path = data_path / "split_meta.json"

    current_hash = _config_hash(
        train_frac, holdout_frac, seed, exclude_prefixes, keep_only_prefixes
    )

    csv_files = sorted(data_path.glob("*.csv"))
    csv_files = [
        f for f in csv_files
        if f.name not in ("holdout.csv", "split_assignment.csv")
    ]

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}")

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df = filter_columns(df, exclude_prefixes, keep_only_prefixes)

    is_stale = True
    if split_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            saved_meta = json.load(f)
        if saved_meta["config_hash"] == current_hash and saved_meta["n_rows"] == len(df):
            is_stale = False

    if is_stale:
        print(
            f"[load_client_data] Nessuno split valido trovato per la config "
            f"attuale (hash={current_hash}) in {data_path}. Rigenero split e holdout."
        )

        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(df))

        n_holdout = int(len(df) * holdout_frac)
        n_train = int((len(df) - n_holdout) * train_frac)

        assignment = np.empty(len(df), dtype=object)
        assignment[perm[:n_holdout]] = "holdout"
        assignment[perm[n_holdout:n_holdout + n_train]] = "train"
        assignment[perm[n_holdout + n_train:]] = "valid"

        pd.DataFrame({"split": assignment}).to_csv(split_path, index=False)
        with open(meta_path, "w") as f:
            json.dump({"config_hash": current_hash, "n_rows": len(df)}, f)
    else:
        assignment = pd.read_csv(split_path)["split"].to_numpy()

    train_df = df[assignment == "train"].reset_index(drop=True)
    valid_df = df[assignment == "valid"].reset_index(drop=True)
    holdout_df = df[assignment == "holdout"].reset_index(drop=True)

    holdout_path = data_path / "holdout.csv"
    if is_stale or not holdout_path.exists():
        holdout_df.to_csv(holdout_path, index=False)

    return (
        dataframe_to_dmatrix(train_df),
        dataframe_to_dmatrix(valid_df),
        len(train_df),
        len(valid_df),
    )

    
def save_holdout(data_path: Path, holdout_df: pd.DataFrame) -> None:
    """Persist the holdout split once, so it stays fixed across rounds."""
    holdout_path = Path(data_path) / "holdout.csv"
    if not holdout_path.exists():
        holdout_df.to_csv(holdout_path, index=False)


def build_centralized_test_set(data_root: Path) -> pd.DataFrame:
    """Concatenate every client's holdout.csv into one test set."""
    data_root = Path(data_root)
    holdout_files = sorted(data_root.glob("*/holdout.csv"))

    if not holdout_files:
        raise ValueError(f"No holdout.csv files found under {data_root}")

    return pd.concat(
        [pd.read_csv(f) for f in holdout_files],
        ignore_index=True,
    )

def filter_columns(
    df: pd.DataFrame,
    exclude_prefixes: list[str] | None = None,
    keep_only_prefixes: list[str] | None = None,
    label_col: str = "label",
) -> pd.DataFrame:
    """Filter dataframe columns by prefix.

    - exclude_prefixes: drop any column whose name starts with one of these prefixes.
    - keep_only_prefixes: keep only columns starting with one of these prefixes
      (the label column is always kept regardless).

    Only one of the two should be set. If both are None, df is returned unchanged.
    """
    if exclude_prefixes and keep_only_prefixes:
        raise ValueError(
            "Set only one of exclude_prefixes or keep_only_prefixes, not both."
        )

    if exclude_prefixes:
        cols_to_drop = [
            c for c in df.columns
            if c != label_col and c.startswith(tuple(exclude_prefixes))
        ]
        return df.drop(columns=cols_to_drop)

    if keep_only_prefixes:
        cols_to_keep = [
            c for c in df.columns
            if c == label_col or c.startswith(tuple(keep_only_prefixes))
        ]
        return df[cols_to_keep]

    return df