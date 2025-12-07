from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Stroke Risk API")


STROKE_LABEL: int = 1

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    "stroke_dt_model.pkl",
    "stroke_dt_model_unpruned.pkl",
    "stroke_dt_model_best.pkl",
    "stroke_model.pkl",
]

MODEL_PATH_USED: str | None = None


class StrokeInput(BaseModel):
    age: float
    gender: str
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str
    smoking_status: str
    avg_glucose_level: float
    bmi: float


def _canon_yes_no(v: str) -> str:
    n = _norm(v)
    return "Yes" if n in {"yes", "y", "true", "1"} else "No"


def _canon_gender(v: str) -> str:
    n = _norm(v)
    if n in {"m", "male"}:
        return "Male"
    if n in {"f", "female"}:
        return "Female"
    return "Other"


def _canon_work_type(v: str) -> str:
    n = _norm(v)
    mapping = {
        "private": "Private",
        "self_employed": "Self-employed",
        "self-employed": "Self-employed",
        "selfemployed": "Self-employed",
        "govt_job": "Govt_job",
        "govt": "Govt_job",
        "government": "Govt_job",
        "children": "children",
        "never_worked": "Never_worked",
        "never-worked": "Never_worked",
        "neverworked": "Never_worked",
    }
    return mapping.get(n, v)


def _canon_residence(v: str) -> str:
    n = _norm(v)
    if n == "urban":
        return "Urban"
    if n == "rural":
        return "Rural"
    return v


def _canon_smoking(v: str) -> str:
    n = _norm(v)
    mapping = {
        "never_smoked": "never smoked",
        "never": "never smoked",
        "formerly_smoked": "formerly smoked",
        "former": "formerly smoked",
        "smokes": "smokes",
        "smoke": "smokes",
        "unknown": "Unknown",
    }
    return mapping.get(n, v)


def _norm(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _json_safe(v: Any) -> Any:
    try:
        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass
    return v


def _unwrap_bundle(obj: Any) -> tuple[Any, list[str], dict]:
    """Support either a raw sklearn model/pipeline or a dict bundle {model, feature_names, ...}."""
    if isinstance(obj, dict) and "model" in obj:
        m = obj["model"]
        fn = list(obj.get("feature_names") or [])
        meta: dict[str, Any] = {}
        for k in ("stroke_label", "positive_label", "target_name", "raw_feature_names"):
            if k in obj:
                meta[k] = obj.get(k)
        return m, fn, meta
    return obj, [], {}


def _model_is_stump(m: Any) -> bool:
    try:
        if hasattr(m, "get_n_leaves"):
            return int(m.get_n_leaves()) <= 1
    except Exception:
        pass
    return False


def _core_estimator(m: Any) -> Any:
    """If this is a Pipeline, try to return the final estimator for depth/leaves."""
    try:
        if hasattr(m, "named_steps") and isinstance(getattr(m, "named_steps", None), dict):
            return m.named_steps.get("model", m)
    except Exception:
        pass
    return m


def _load_first_good_model() -> tuple[Any, list[str], dict]:
    global MODEL_PATH_USED
    last_model = None
    last_feature_names: list[str] = []
    last_meta: dict = {}

    for fname in MODEL_CANDIDATES:
        path = BASE_DIR / fname
        if not path.exists():
            continue

        obj = joblib.load(path)
        m, fn, meta = _unwrap_bundle(obj)
        last_model, last_feature_names, last_meta = m, fn, meta
        MODEL_PATH_USED = str(path)

        core = _core_estimator(m)
        if not _model_is_stump(core):
            return m, fn, meta

    if last_model is None:
        raise FileNotFoundError(
            f"No model file found. Expected one of: {', '.join(MODEL_CANDIDATES)} in {BASE_DIR}"
        )
    return last_model, last_feature_names, last_meta


model, feature_names, model_meta = _load_first_good_model()

if isinstance(model_meta, dict):
    for key in ("stroke_label", "positive_label"):
        if key in model_meta and model_meta[key] is not None:
            try:
                STROKE_LABEL = int(model_meta[key])
                break
            except Exception:
                pass

if not feature_names and hasattr(model, "feature_names_in_"):
    try:
        feature_names = list(model.feature_names_in_)
    except Exception:
        feature_names = []


def _onehot(cols: list[str]) -> bool:
    return any(
        str(c).lower().startswith(("gender_", "work_type_", "residence_type_", "smoking_status_"))
        for c in cols
    )


def _classes_safe(m: Any) -> list[Any]:
    raw = getattr(m, "classes_", [])
    return [ _json_safe(c) for c in list(raw) ]


def _is_pipeline_like(m: Any) -> bool:
    return hasattr(m, "named_steps") and hasattr(m, "predict") and hasattr(m, "predict_proba")


def _raw_row_for_pipeline(x: StrokeInput, raw_cols: list[str]) -> dict[str, Any]:
    cols = set(raw_cols)
    residence_key = "Residence_type" if "Residence_type" in cols else ("residence_type" if "residence_type" in cols else "Residence_type")

    row: dict[str, Any] = {
        "age": float(x.age),
        "gender": _canon_gender(x.gender),
        "hypertension": int(x.hypertension),
        "heart_disease": int(x.heart_disease),
        "ever_married": _canon_yes_no(x.ever_married),
        "work_type": _canon_work_type(x.work_type),
        residence_key: _canon_residence(x.residence_type),
        "smoking_status": _canon_smoking(x.smoking_status),
        "avg_glucose_level": float(x.avg_glucose_level),
        "bmi": float(x.bmi),
    }
    if raw_cols:
        row = {k: row.get(k) for k in raw_cols}
    return row


def build_X(x: StrokeInput) -> np.ndarray:
    """Manual one-hot builder for legacy models that were trained on already-one-hot columns."""
    if not feature_names:
        raise RuntimeError(
            "feature_names missing in model bundle. Re-train and save a pipeline bundle, or save feature_names with the model."
        )

    if _onehot(feature_names):
        g = _norm(x.gender)
        w = _norm(x.work_type)
        r = _norm(x.residence_type)
        s = _norm(x.smoking_status)

        vals: list[float] = []
        for col in feature_names:
            c = _norm(str(col))

            if c == "age":
                vals.append(float(x.age))
            elif c in ("avg_glucose_level", "avg_glucose"):
                vals.append(float(x.avg_glucose_level))
            elif c == "bmi":
                vals.append(float(x.bmi))
            elif c == "hypertension":
                vals.append(float(int(x.hypertension)))
            elif c == "heart_disease":
                vals.append(float(int(x.heart_disease)))
            elif c == "ever_married":
                vals.append(1.0 if _norm(x.ever_married) == "yes" else 0.0)
            elif c.startswith("gender_"):
                vals.append(1.0 if g == c[len("gender_"):] else 0.0)
            elif c.startswith("work_type_"):
                vals.append(1.0 if w == c[len("work_type_"):] else 0.0)
            elif c.startswith("residence_type_"):
                vals.append(1.0 if r == c[len("residence_type_"):] else 0.0)
            elif c.startswith("smoking_status_"):
                vals.append(1.0 if s == c[len("smoking_status_"):] else 0.0)
            else:
                vals.append(0.0)

        return np.array(vals, dtype=float).reshape(1, -1)

    vals: list[float] = []
    for col in feature_names:
        c = _norm(str(col))
        if c == "age":
            vals.append(float(x.age))
        elif c in ("avg_glucose_level", "avg_glucose"):
            vals.append(float(x.avg_glucose_level))
        elif c == "bmi":
            vals.append(float(x.bmi))
        elif c == "hypertension":
            vals.append(float(int(x.hypertension)))
        elif c == "heart_disease":
            vals.append(float(int(x.heart_disease)))
        elif c == "ever_married":
            vals.append(1.0 if _norm(x.ever_married) == "yes" else 0.0)
        else:
            vals.append(0.0)
    return np.array(vals, dtype=float).reshape(1, -1)


def summarize_features(X: np.ndarray) -> dict[str, Any]:
    """Return which one-hot columns are active, plus key numeric features."""
    if X.ndim != 2 or X.shape[0] != 1:
        return {"active_onehots": [], "numeric": {}}

    row = X[0]
    active_onehots: list[str] = []
    numeric: dict[str, float] = {}

    for name, val in zip(feature_names, row):
        n = str(name)
        lc = _norm(n)
        fv = float(val)

        if lc.startswith(("gender_", "work_type_", "residence_type_", "smoking_status_")):
            if fv >= 0.5:
                active_onehots.append(n)
        elif lc in ("age", "bmi", "avg_glucose_level", "avg_glucose", "hypertension", "heart_disease", "ever_married"):
            numeric[n] = fv

    return {"active_onehots": active_onehots, "numeric": numeric}


def _predict_proba_any(m: Any, x: StrokeInput) -> tuple[list[Any], list[float]]:
    """Return (classes, probs) for a single input, handling pipeline or manual one-hot models."""
    if not hasattr(m, "predict_proba"):
        classes = _classes_safe(m)
        if not classes:
            classes = [0, 1]
        pred = 0
        try:
            pred = int(_json_safe(m.predict(build_X(x))[0]))
        except Exception:
            pass
        probs = [0.0 for _ in classes]
        try:
            probs[classes.index(pred)] = 1.0
        except Exception:
            pass
        return classes, probs

    if _is_pipeline_like(m) and hasattr(m, "feature_names_in_"):
        raw_cols = list(getattr(m, "feature_names_in_", []))
        row = _raw_row_for_pipeline(x, raw_cols)
        try:
            import pandas as pd

            df = pd.DataFrame([row])
            probs = m.predict_proba(df)[0]
        except Exception:
            ordered = [row[c] for c in raw_cols]
            probs = m.predict_proba(np.array(ordered, dtype=object).reshape(1, -1))[0]
        classes = _classes_safe(m)
        return classes, [float(p) for p in probs]

    X = build_X(x)
    probs = m.predict_proba(X)[0]
    classes = _classes_safe(m)
    return classes, [float(p) for p in probs]


@app.get("/health")
def health():
    core = _core_estimator(model)

    depth = None
    leaves = None
    try:
        if hasattr(core, "get_depth"):
            depth = int(core.get_depth())
        if hasattr(core, "get_n_leaves"):
            leaves = int(core.get_n_leaves())
    except Exception:
        pass

    sanity = None
    try:
        low = StrokeInput(
            age=23,
            gender="Male",
            hypertension=0,
            heart_disease=0,
            ever_married="No",
            work_type="Private",
            residence_type="Urban",
            smoking_status="never smoked",
            avg_glucose_level=95,
            bmi=22,
        )
        high = StrokeInput(
            age=78,
            gender="Male",
            hypertension=1,
            heart_disease=1,
            ever_married="Yes",
            work_type="Self-employed",
            residence_type="Urban",
            smoking_status="smokes",
            avg_glucose_level=210,
            bmi=36,
        )

        cls_low, p_low = _predict_proba_any(model, low)
        cls_high, p_high = _predict_proba_any(model, high)

        def _map(cls, probs):
            return {str(_json_safe(cls[i])): float(probs[i]) for i in range(min(len(cls), len(probs)))}

        sanity = {
            "classes_low": [str(_json_safe(c)) for c in cls_low],
            "probs_low": _map(cls_low, p_low),
            "classes_high": [str(_json_safe(c)) for c in cls_high],
            "probs_high": _map(cls_high, p_high),
        }
    except Exception:
        sanity = None

    return {
        "status": "ok",
        "model_path": MODEL_PATH_USED,
        "model_type": type(model).__name__,
        "model_depth": depth,
        "model_leaves": leaves,
        "stroke_label": STROKE_LABEL,
        "n_features": len(feature_names),
        "has_predict_proba": bool(hasattr(model, "predict_proba")),
        "onehot": _onehot(feature_names),
        "classes": [str(_json_safe(c)) for c in _classes_safe(model)],
        "sanity": sanity,
    }


@app.post("/predict")
def predict(x: StrokeInput):
    classes, probs = _predict_proba_any(model, x)

    pred = int(_json_safe(classes[int(np.argmax(probs))])) if classes else 0

    if STROKE_LABEL in classes:
        stroke_prob = float(probs[classes.index(STROKE_LABEL)])
    elif len(probs) >= 2:
        stroke_prob = float(probs[1])
    elif len(probs) == 1:
        stroke_prob = float(probs[0])
    else:
        stroke_prob = 0.0

    prob_map = {str(_json_safe(classes[i])): float(probs[i]) for i in range(min(len(classes), len(probs)))}

    feature_debug: dict[str, Any]
    if _is_pipeline_like(model):
        raw_cols = list(getattr(model, "feature_names_in_", []))
        feature_debug = {"raw": _raw_row_for_pipeline(x, raw_cols)}
    else:
        X = build_X(x)
        feature_debug = summarize_features(X)

    print("[api] input:", x.model_dump())
    print("[api] classes:", [str(_json_safe(c)) for c in classes])
    print("[api] STROKE_LABEL:", STROKE_LABEL)
    print("[api] pred:", pred, "stroke_prob:", stroke_prob)
    print("[api] probs:", prob_map)

    prob_pct = stroke_prob * 100.0
    risk_level = "low" if prob_pct <= 33 else ("medium" if prob_pct <= 66 else "high")

    return {
        "prediction": pred,
        "stroke_label": STROKE_LABEL,
        "classes": [ _json_safe(c) for c in classes ],
        "probabilities": prob_map,
        "stroke_probability": float(stroke_prob),
        "risk_level": risk_level,
        "feature_debug": feature_debug,
    }