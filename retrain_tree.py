""" To run this file 
Terminal 1 : 
# 1) Go to project root
cd /Users/mohammedmubashiruddinfaraz/Documents/Machine_Learning/Decision_Tree

# 2) Create venv (first time only)
python3 -m venv .venv

# 3) Activate venv
source .venv/bin/activate

# 4) Install dependencies (first time only)
pip install fastapi "uvicorn[standard]" joblib numpy

# 5) Run backend server
uvicorn stroke_api:app --reload --host 127.0.0.1 --port 8000

Terminal 2 :
# 1) Go to the website folder
cd /Users/mohammedmubashiruddinfaraz/Documents/Machine_Learning/Decision_Tree/stroke-risk-prediction-app

# 2) Install node deps (first time only)
npm install

# 3) Run the dev server
npm run dev
"""




from __future__ import annotations

import argparse

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

UI_FEATURES = [
    "age",
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "residence_type",
    "smoking_status",
    "avg_glucose_level",
    "bmi",
]

CAT_COLS = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]

_SMOKING_CANON = {
    "never smoked": "Never smoked",
    "formerly smoked": "Formerly smoked",
    "smokes": "Smokes",
    "unknown": "Unknown",
}


def _is_usable_dataset(p: Path) -> bool:
    """True iff CSV contains stroke + all UI feature columns (after column normalization)."""
    try:
        head = pd.read_csv(p, nrows=50)
        head = _normalize_columns(head)
        cols = set(head.columns)
        needed = set(["stroke"] + UI_FEATURES)
        return needed.issubset(cols)
    except Exception:
        return False


def _oversample_minority(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    target_pos_ratio: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Randomly oversample the positive class to a target ratio in the training set.

    This helps a plain DecisionTree actually learn the minority class (stroke=1).
    Note: oversampling changes probability calibration; use probabilities as a relative score.
    """
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    pos_idx = y[y == 1].index.to_numpy()
    neg_idx = y[y == 0].index.to_numpy()

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return X, y

    desired_pos = int(np.ceil((target_pos_ratio * n_neg) / max(1e-9, (1.0 - target_pos_ratio))))
    if desired_pos <= n_pos:
        return X, y

    rng = np.random.default_rng(random_state)
    extra = rng.choice(pos_idx, size=(desired_pos - n_pos), replace=True)

    all_idx = np.concatenate([neg_idx, pos_idx, extra])
    rng.shuffle(all_idx)

    return X.iloc[all_idx].reset_index(drop=True), y.iloc[all_idx].reset_index(drop=True)


def _find_dataset(base_dir: Path, provided: str | None = None) -> Path:
    """Pick a CSV that contains 'stroke' and the raw UI feature columns.

    If `provided` is given, use that path.
    Otherwise, search common filenames in `base_dir`, then search recursively.
    """
    if provided:
        p = Path(provided).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Provided dataset path does not exist: {p}")
        if not _is_usable_dataset(p):
            raise ValueError(
                "Provided dataset is not usable for this UI/API. It must contain 'stroke' + these columns: "
                + ", ".join(UI_FEATURES)
                + f". Got file: {p}"
            )
        return p

    candidates = [
        base_dir / "healthcare-dataset-stroke-data.csv",
        base_dir / "healthcare_dataset_stroke_data.csv",
        base_dir / "stroke.csv",
        base_dir / "strokes.csv",
        base_dir / "dataset.csv",
    ]

    for p in candidates:
        if p.exists() and _is_usable_dataset(p):
            return p

    for p in sorted(base_dir.glob("*.csv")):
        if _is_usable_dataset(p):
            return p

    for p in sorted(base_dir.rglob("*.csv")):
        name = p.name.lower()
        if "preprocess" in name or "preprocessed" in name or "encoded" in name or "onehot" in name:
            continue
        if _is_usable_dataset(p):
            return p

   
    parent = base_dir.parent
    for p in sorted(parent.rglob("*.csv")):
        name = p.name.lower()
        if "preprocess" in name or "preprocessed" in name or "encoded" in name or "onehot" in name:
            continue
        if _is_usable_dataset(p):
            return p

    raise FileNotFoundError(
        f"No suitable CSV found in {base_dir}. Need a CSV with 'stroke' + these columns: "
        + ", ".join(UI_FEATURES)
        + "\nTip 1: run `find .. -name \"*.csv\" | grep -i stroke` to locate your raw dataset."
        + "\nTip 2: then run `python retrain_tree.py --data \"/absolute/path/to/your.csv\"` (quote it if it has spaces)."
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Residence_type" in df.columns and "residence_type" not in df.columns:
        df = df.rename(columns={"Residence_type": "residence_type"})
    return df


def _prep(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = _normalize_columns(df.copy())

    for c in ["id", "ID", "Id"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    if "stroke" not in df.columns:
        raise ValueError("Dataset must contain a 'stroke' column.")

    y = df["stroke"].astype(int)

    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    for col in ["age", "avg_glucose_level"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    keep = UI_FEATURES

    X = df[keep].copy()

    for c in CAT_COLS:
        X[c] = X[c].astype(str).str.strip()

    X["smoking_status"] = (
        X["smoking_status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(_SMOKING_CANON)
        .fillna(X["smoking_status"])
    )

    X["hypertension"] = X["hypertension"].astype(int)
    X["heart_disease"] = X["heart_disease"].astype(int)

    cat_cols = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    for col in ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

    return X, y


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain stroke DecisionTree model for the web UI/API")
    parser.add_argument(
        "--data",
        default=None,
        help="Path to a raw stroke CSV containing 'stroke' and the UI columns (age, gender, work_type, etc.)",
    )
    parser.add_argument(
        "--target-pos-ratio",
        type=float,
        default=0.20,
        help="Target positive (stroke=1) ratio after oversampling in the TRAIN split (default: 0.20)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_dir = Path(__file__).resolve().parent
    data_path = _find_dataset(base_dir, provided=args.data)
    print(f"Using dataset: {data_path}")

    df = pd.read_csv(data_path)
    X, y = _prep(df)

    print("\nClass distribution (counts):")
    print(y.value_counts().rename_axis("stroke").to_string())
    print("\nClass distribution (percent):")
    print((y.value_counts(normalize=True) * 100).round(2).rename_axis("stroke").to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    X_train_os, y_train_os = _oversample_minority(
        X_train,
        y_train,
        target_pos_ratio=float(args.target_pos_ratio),
        random_state=42,
    )
    pos_ratio = float((y_train_os == 1).mean())
    print(f"\nTraining positive ratio after oversampling: {pos_ratio:.3f}")

    base = DecisionTreeClassifier(random_state=42, class_weight="balanced")

    param_grid = {
        "max_depth": [3, 4, 5, 6],
        "min_samples_leaf": [10, 20, 40, 80],
        "min_samples_split": [20, 50, 100],
        "ccp_alpha": [0.0, 0.0005, 0.001, 0.003],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="average_precision",
        n_jobs=-1,
        cv=cv,
        refit=True,
        verbose=0,
    )
    gs.fit(X_train_os, y_train_os)
    model: DecisionTreeClassifier = gs.best_estimator_

    print("\nBest params:")
    print(gs.best_params_)

    proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    pred = (proba >= 0.5).astype(int)

    print("\nConfusion matrix (threshold=0.50):")
    print(confusion_matrix(y_test, pred))

    print("\nClassification report (threshold=0.50):")
    print(classification_report(y_test, pred, digits=2))

    auc = roc_auc_score(y_test, proba)
    print(f"ROC-AUC (label 1): {auc:.6f}") 

    bundle = {
        "model": model,
        "feature_names": list(X.columns),
        "stroke_label": 1,
        "target_name": "stroke",
        "raw_feature_names": [
            "age","gender","hypertension","heart_disease","ever_married",
            "work_type","residence_type","smoking_status","avg_glucose_level","bmi",
        ],
        "preprocess": {
            "smoking_canon": _SMOKING_CANON,
        },
        "notes": {
            "oversample_target_pos_ratio": float(args.target_pos_ratio),
            "grid_scoring": "average_precision",
        },
    }

    out_path = base_dir / "stroke_dt_model.pkl"
    joblib.dump(bundle, out_path)
    print(f"Saved {out_path}")

    low_row = {
        "age": 25, "gender": "Male", "hypertension": 0, "heart_disease": 0,
        "ever_married": "No", "work_type": "Private", "residence_type": "Urban",
        "smoking_status": "Never smoked", "avg_glucose_level": 95, "bmi": 22,
    }
    high_row = {
        "age": 78, "gender": "Male", "hypertension": 1, "heart_disease": 1,
        "ever_married": "Yes", "work_type": "Self-employed", "residence_type": "Urban",
        "smoking_status": "Smokes", "avg_glucose_level": 210, "bmi": 36,
    }

    low_row["smoking_status"] = _SMOKING_CANON.get(str(low_row["smoking_status"]).strip().lower(), low_row["smoking_status"])
    high_row["smoking_status"] = _SMOKING_CANON.get(str(high_row["smoking_status"]).strip().lower(), high_row["smoking_status"])

    low_df = pd.get_dummies(pd.DataFrame([low_row]),
                            columns=["gender","ever_married","work_type","residence_type","smoking_status"],
                            drop_first=False)
    high_df = pd.get_dummies(pd.DataFrame([high_row]),
                             columns=["gender","ever_married","work_type","residence_type","smoking_status"],
                             drop_first=False)

    low_df = low_df.reindex(columns=X.columns, fill_value=0.0)
    high_df = high_df.reindex(columns=X.columns, fill_value=0.0)

    p_low = model.predict_proba(low_df)[:, list(model.classes_).index(1)][0]
    p_high = model.predict_proba(high_df)[:, list(model.classes_).index(1)][0]
    print(f"\nSanity stroke probability: low={p_low:.3f} high={p_high:.3f}")


if __name__ == "__main__":
    main()