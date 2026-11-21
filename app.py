"""
SALARY PREDICTION STUDIO
Intelligent Salary Estimation Powered by Machine Learning.

Complete single-file Python / Streamlit application (`app.py`).
No external HTML, CSS, JavaScript, component folders, or separate Python modules.

Dependencies:
    pip install streamlit pandas numpy scikit-learn xgboost plotly joblib

Run:
    streamlit run app.py

Dataset:
    Place `employee_salary_dataset.csv` in the same folder as `app.py`, or upload it
    directly within the single-page interface when prompted.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Suppress bare-mode ScriptRunContext warnings when imported or invoked via `python app.py`
for _logger_name in (
    "streamlit.runtime.scriptrunner_utils.script_run_context",
    "streamlit.runtime.caching.cache_data_api",
    "streamlit.runtime.state.session_state_proxy",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBRegressor = None  # type: ignore[assignment]
    XGBOOST_AVAILABLE = False


# =============================================================================
# CONSTANTS & EXACT NOTEBOOK FEATURE SPECIFICATION
# =============================================================================

APP_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd().resolve()
)
APP_FILE = (
    Path(__file__).resolve()
    if "__file__" in globals()
    else (APP_DIR / "app.py").resolve()
)
DEFAULT_DATASET_PATH = APP_DIR / "employee_salary_dataset.csv"
BACKEND_PKL_PATH = APP_DIR / "employee_selary.pkl"
PIPELINE_SCHEMA_VERSION = "salary-prediction-studio-backend-pkl-v4"

# Allowed employee age range for the Age dial and prediction input validation
EMPLOYEE_AGE_MIN = 18
EMPLOYEE_AGE_MAX = 60

TARGET_COLUMN = "Annual_Salary_LPA"
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Education",
    "Experience_Years",
    "Department",
    "Job_Level",
    "Remote_Work",
    "City",
    "Skill_Score",
]
NOTEBOOK_MODEL_COLUMNS = [
    "Age",
    "Gender",
    "Education",
    "Experience_Years",
    "Department",
    "Job_Level",
    "Performance_Rating",
    "Certifications",
    "Overtime_Hours",
    "Remote_Work",
    "City",
    "Company_Tenure",
    "Projects_Completed",
    "Skill_Score",
]
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Exact ordinal mappings preserved from employee_selary.ipynb (Cells 36-46):
# Gender:      Male=0, Female=1
# Education:   Diploma=0, Bachelor=1, Master=2, PhD=3
# Department:  Operations=0, Marketing=1, Sales=2, IT=3, HR=4
# Job_Level:   Junior=0, Lead=1, Mid=2, Senior=3, Manager=4
# Remote_Work: No=0, Yes=1
# City:        Chennai=0, Delhi=1, Hyderabad=2, Mumbai=3
NOTEBOOK_CATEGORY_MAPS: dict[str, dict[str, int]] = {
    "Gender": {"Male": 0, "Female": 1},
    "Education": {"Diploma": 0, "Bachelor": 1, "Master": 2, "PhD": 3},
    "Department": {"Operations": 0, "Marketing": 1, "Sales": 2, "IT": 3, "HR": 4},
    "Job_Level": {"Junior": 0, "Lead": 1, "Mid": 2, "Senior": 3, "Manager": 4},
    "Remote_Work": {"No": 0, "Yes": 1},
    "City": {"Chennai": 0, "Delhi": 1, "Hyderabad": 2, "Mumbai": 3},
}

CATEGORICAL_ENCODING_ORDER: dict[str, list[str]] = {
    "Gender": ["Male", "Female"],
    "Education": ["Diploma", "Bachelor", "Master", "PhD"],
    "Department": ["Operations", "Marketing", "Sales", "IT", "HR"],
    "Job_Level": ["Junior", "Lead", "Mid", "Senior", "Manager"],
    "Remote_Work": ["No", "Yes"],
    "City": ["Chennai", "Delhi", "Hyderabad", "Mumbai"],
}

UI_CATEGORY_OPTIONS: dict[str, list[str]] = {
    "Gender": ["Male", "Female"],
    "Education": ["Master", "Bachelor", "Diploma", "PhD"],
    "Department": ["Operations", "Marketing", "Sales", "IT", "HR"],
    "Job_Level": ["Junior", "Lead", "Mid", "Senior", "Manager"],
    "Remote_Work": ["Yes", "No"],
    "City": ["Chennai", "Delhi", "Hyderabad", "Mumbai"],
}

# Exact sample profile from Cell 82 (data_new / EMP10001) of employee_selary.ipynb
NOTEBOOK_CELL82_DEFAULTS: dict[str, Any] = {
    "Age": 28,
    "Gender": "Male",
    "Education": "Master",
    "Experience_Years": 6.0,
    "Department": "Operations",
    "Job_Level": "Junior",
    "Performance_Rating": 1,
    "Certifications": 4,
    "Overtime_Hours": 15,
    "Remote_Work": "Yes",
    "City": "Chennai",
    "Company_Tenure": 6,
    "Projects_Completed": 5,
    "Skill_Score": 97.0,
}


# =============================================================================
# DATASET VALIDATION & METADATA EXTRACTION (FROM employee_selary.ipynb)
# =============================================================================


def compute_dataset_hash(data: bytes) -> str:
    hasher = hashlib.sha256()
    hasher.update(PIPELINE_SCHEMA_VERSION.encode("utf-8"))
    hasher.update(str(XGBOOST_AVAILABLE).encode("utf-8"))
    hasher.update(data)
    return hasher.hexdigest()


def validate_and_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns, numeric columns, missing values, and exact categories."""
    if df is None or df.empty:
        raise ValueError("The uploaded dataset CSV is empty.")

    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in cleaned.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required column(s): {', '.join(missing_cols)}."
        )

    if len(cleaned) < 15:
        raise ValueError(
            "Dataset must contain at least 15 valid rows for an 80/20 train-test split."
        )

    for num_col in ["Age", "Experience_Years", "Skill_Score", TARGET_COLUMN]:
        raw_col = cleaned[num_col]
        present_mask = raw_col.notna() & (raw_col.astype(str).str.strip() != "")
        numeric_col = pd.to_numeric(raw_col, errors="coerce")
        bad_mask = present_mask & numeric_col.isna()
        if bad_mask.any():
            samples = raw_col[bad_mask].astype(str).unique()[:3].tolist()
            raise ValueError(
                f"Column '{num_col}' contains invalid numeric value(s): {', '.join(samples)}."
            )
        cleaned[num_col] = numeric_col

    valid_target = cleaned[TARGET_COLUMN].notna() & (cleaned[TARGET_COLUMN] > 0)
    if not valid_target.all():
        cleaned = cleaned.loc[valid_target].copy()

    # Exact notebook Cell 28 filter: data = data[~(data['Age']>=61.0) & (data['Age']<=90.0)]
    cleaned = cleaned[~(cleaned["Age"] >= 61.0) & (cleaned["Age"] <= 90.0)].copy()

    if len(cleaned) < 15:
        raise ValueError(
            "Fewer than 15 valid rows remain after removing rows with invalid target salaries."
        )

    for num_col in ["Age", "Experience_Years", "Skill_Score"]:
        if cleaned[num_col].notna().sum() == 0:
            raise ValueError(f"Column '{num_col}' contains no valid numeric values.")

    for cat_col, allowed_values in CATEGORICAL_ENCODING_ORDER.items():
        if cleaned[cat_col].isna().any():
            raise ValueError(f"Categorical column '{cat_col}' contains missing values.")
        cleaned[cat_col] = cleaned[cat_col].astype(str).str.strip()
        observed = set(cleaned[cat_col].unique())
        unexpected = sorted(observed - set(allowed_values))
        if unexpected:
            raise ValueError(
                f"Column '{cat_col}' contains unsupported categor(ies): {', '.join(unexpected)}. "
                f"Expected: {', '.join(allowed_values)}."
            )

    return cleaned


def extract_dataset_metadata(df: pd.DataFrame, source_name: str) -> dict[str, Any]:
    """Derive actual min/max bounds, defaults, auxiliary medians, and target statistics from the dataset."""
    valid_age = df["Age"].dropna()
    valid_exp = df["Experience_Years"].dropna()
    valid_skill = df["Skill_Score"].dropna()
    target = df[TARGET_COLUMN]

    age_min = EMPLOYEE_AGE_MIN
    age_max = EMPLOYEE_AGE_MAX
    age_default = int(np.clip(round(float(valid_age.mean())), age_min, age_max))

    exp_has_decimals = bool(((valid_exp % 1).abs() > 1e-6).any())
    exp_step = 0.5 if exp_has_decimals else 1.0
    exp_min = float(math.floor(float(valid_exp.min()) / exp_step) * exp_step)
    exp_max = float(math.ceil(float(valid_exp.max()) / exp_step) * exp_step)
    exp_default = float(
        np.clip(round(float(valid_exp.median()) / exp_step) * exp_step, exp_min, exp_max)
    )

    skill_has_decimals = bool(((valid_skill % 1).abs() > 1e-6).any())
    skill_step = 0.5 if skill_has_decimals else 1.0
    skill_min = float(math.floor(float(valid_skill.min()) / skill_step) * skill_step)
    skill_max = float(math.ceil(float(valid_skill.max()) / skill_step) * skill_step)
    skill_default = float(
        np.clip(
            round(float(valid_skill.median()) / skill_step) * skill_step,
            skill_min,
            skill_max,
        )
    )

    aux_defaults = {
        "Performance_Rating": int(round(float(df["Performance_Rating"].median())))
        if "Performance_Rating" in df.columns
        else 3,
        "Certifications": int(round(float(df["Certifications"].median())))
        if "Certifications" in df.columns
        else 5,
        "Overtime_Hours": int(round(float(df["Overtime_Hours"].median())))
        if "Overtime_Hours" in df.columns
        else 30,
        "Company_Tenure": int(round(float(df["Company_Tenure"].median())))
        if "Company_Tenure" in df.columns
        else 5,
        "Projects_Completed": int(round(float(df["Projects_Completed"].median())))
        if "Projects_Completed" in df.columns
        else 15,
    }

    return {
        "source_name": source_name,
        "row_count": int(len(df)),
        "missing_counts": {
            "Age": int(df["Age"].isna().sum()),
            "Experience_Years": int(df["Experience_Years"].isna().sum()),
            "Skill_Score": int(df["Skill_Score"].isna().sum()),
        },
        "average_salary_lpa": round(float(target.mean()), 2),
        "min_salary_lpa": round(float(target.min()), 2),
        "max_salary_lpa": round(float(target.max()), 2),
        "aux_defaults": aux_defaults,
        "bounds": {
            "Age": {
                "min": age_min,
                "max": age_max,
                "default": age_default,
                "step": 1,
                "is_int": True,
            },
            "Experience_Years": {
                "min": exp_min,
                "max": exp_max,
                "default": exp_default,
                "step": exp_step,
                "is_int": not exp_has_decimals,
            },
            "Skill_Score": {
                "min": skill_min,
                "max": skill_max,
                "default": skill_default,
                "step": skill_step,
                "is_int": not skill_has_decimals,
            },
        },
    }


# =============================================================================
# BACKEND CONNECTION: employee_selary.pkl & NOTEBOOK PREPROCESSING PIPELINE
# =============================================================================


def preprocess_notebook_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact preprocessing from employee_selary.ipynb (Cells 36-53)."""
    data = df.copy()
    for cat_col, mapping in NOTEBOOK_CATEGORY_MAPS.items():
        data[cat_col] = data[cat_col].map(mapping)

    data["Age"] = data["Age"].fillna(data["Age"].mean())
    data["Experience_Years"] = data["Experience_Years"].fillna(data["Experience_Years"].median())
    data["Skill_Score"] = data["Skill_Score"].fillna(data["Skill_Score"].median())
    return data


def load_backend_pickle_model(pkl_path: Path = BACKEND_PKL_PATH) -> Any | None:
    """Load the pre-trained model from employee_selary.pkl using Joblib."""
    if not pkl_path.exists():
        return None
    try:
        return joblib.load(pkl_path)
    except Exception:
        return None


def train_and_select_best_model(
    df: pd.DataFrame,
    dataset_hash: str,
    source_name: str,
) -> dict[str, Any]:
    """Connect to employee_selary.pkl and evaluate all 4 notebook regression models on the 80/20 split."""
    encoded_df = preprocess_notebook_dataframe(df)
    drop_cols = [c for c in ["Employee_ID", TARGET_COLUMN] if c in encoded_df.columns]
    X_full = encoded_df.drop(drop_cols, axis=1)
    y = encoded_df[TARGET_COLUMN].astype(float).copy()

    # Exact notebook Cell 58 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.20, random_state=42
    )

    # 1. Load pre-trained backend model from employee_selary.pkl
    backend_pkl_model = load_backend_pickle_model(BACKEND_PKL_PATH)
    pkl_connected = backend_pkl_model is not None

    # 2. Train the 4 benchmark regression algorithms from employee_selary.ipynb (Cell 63)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    gbr_model = GradientBoostingRegressor(random_state=42)
    gbr_model.fit(X_train, y_train)

    candidate_eval_models: list[tuple[str, Any, Any]] = [
        ("Linear Regression", lr_model, lr_model),
        ("Random Forest Regressor", rf_model, rf_model),
        ("Gradient Boosting Regressor", gbr_model, gbr_model),
    ]

    if XGBOOST_AVAILABLE and XGBRegressor is not None:
        xgb_split = XGBRegressor(random_state=42, n_jobs=-1)
        xgb_split.fit(X_train, y_train)
        if backend_pkl_model is None:
            backend_pkl_model = XGBRegressor(random_state=42, n_jobs=-1)
            backend_pkl_model.fit(X_full, y)
            try:
                joblib.dump(backend_pkl_model, BACKEND_PKL_PATH)
                pkl_connected = True
            except Exception:
                pass
        candidate_eval_models.append(
            ("XGBoost Regressor (employee_selary.pkl)", xgb_split, backend_pkl_model)
        )
    elif backend_pkl_model is not None:
        candidate_eval_models.append(
            ("XGBoost Regressor (employee_selary.pkl)", backend_pkl_model, backend_pkl_model)
        )

    evaluations: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}

    for model_name, eval_model, infer_model in candidate_eval_models:
        y_pred = eval_model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_test, y_pred)))

        fitted_models[model_name] = infer_model
        evaluations.append(
            {
                "model_name": model_name,
                "r2_score": round(r2, 4),
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "is_selected": model_name == "XGBoost Regressor (employee_selary.pkl)",
            }
        )

    # Default active backend model is employee_selary.pkl when available, else lowest RMSE
    primary_name = (
        "XGBoost Regressor (employee_selary.pkl)"
        if "XGBoost Regressor (employee_selary.pkl)" in fitted_models
        else min(evaluations, key=lambda item: item["rmse"])["model_name"]
    )
    for ev in evaluations:
        ev["is_selected"] = ev["model_name"] == primary_name
    selected_eval = next(ev for ev in evaluations if ev["model_name"] == primary_name)

    dataset_meta = extract_dataset_metadata(df, source_name)
    dataset_meta["train_count"] = int(len(X_train))
    dataset_meta["test_count"] = int(len(X_test))

    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "dataset_hash": dataset_hash,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "xgboost_available": XGBOOST_AVAILABLE,
        "pkl_connected": pkl_connected,
        "pkl_filename": BACKEND_PKL_PATH.name,
        "selected_model_name": primary_name,
        "selected_metrics": selected_eval,
        "all_evaluations": evaluations,
        "models": fitted_models,
        "pipeline": fitted_models[primary_name],
        "dataset_meta": dataset_meta,
    }


def load_or_train_artifact(csv_bytes: bytes, source_name: str) -> dict[str, Any]:
    """Load dataset and connect directly to employee_selary.pkl and the 4 notebook regressors."""
    dataset_hash = compute_dataset_hash(csv_bytes)
    raw_df = pd.read_csv(io.BytesIO(csv_bytes))
    clean_df = validate_and_clean_dataframe(raw_df)
    return train_and_select_best_model(clean_df, dataset_hash, source_name)


# =============================================================================
# CENTRAL UI-INDEPENDENT PREDICTION FUNCTION (CONNECTED TO employee_selary.pkl)
# =============================================================================


def _encode_notebook_category(val: Any, col_name: str) -> int:
    mapping = NOTEBOOK_CATEGORY_MAPS[col_name]
    if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        if int(val) in mapping.values():
            return int(val)
    if val is None:
        raise ValueError(f"Feature '{col_name}' cannot be empty.")
    cat_str = str(val).strip()
    if cat_str in mapping:
        return mapping[cat_str]
    raise ValueError(
        f"Unexpected category '{cat_str}' for '{col_name}'. Allowed: {', '.join(mapping.keys())}."
    )


def predict_salary(
    employee_details: dict[str, Any],
    artifact: dict[str, Any] | None = None,
) -> float:
    """Validate employee inputs, apply employee_selary.ipynb preprocessing, and predict via employee_selary.pkl."""
    if not isinstance(employee_details, dict):
        raise TypeError("employee_details must be a dictionary of employee feature values.")

    if artifact is None and DEFAULT_DATASET_PATH.exists():
        artifact = load_or_train_artifact(
            DEFAULT_DATASET_PATH.read_bytes(), DEFAULT_DATASET_PATH.name
        )

    missing_features = [col for col in FEATURE_COLUMNS if col not in employee_details]
    if missing_features:
        raise ValueError(
            f"Missing required employee feature(s): {', '.join(missing_features)}."
        )

    bounds = (
        artifact["dataset_meta"]["bounds"]
        if artifact is not None
        else {
            "Age": {"min": EMPLOYEE_AGE_MIN, "max": EMPLOYEE_AGE_MAX},
            "Experience_Years": {"min": 0.0, "max": 33.0},
            "Skill_Score": {"min": 50.0, "max": 100.0},
        }
    )
    aux_defaults = (
        artifact["dataset_meta"].get("aux_defaults", {})
        if artifact is not None
        else {
            "Performance_Rating": 3,
            "Certifications": 5,
            "Overtime_Hours": 30,
            "Company_Tenure": 5,
            "Projects_Completed": 15,
        }
    )

    validated_num: dict[str, float] = {}
    for num_col in ["Age", "Experience_Years", "Skill_Score"]:
        val = employee_details[num_col]
        if val is None or isinstance(val, bool):
            raise ValueError(f"Feature '{num_col}' must be a valid number.")
        try:
            num_val = float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{num_col}' must be numeric.") from exc

        if not math.isfinite(num_val):
            raise ValueError(f"Feature '{num_col}' must be finite.")

        lo = float(bounds[num_col]["min"])
        hi = float(bounds[num_col]["max"])
        if num_val < lo or num_val > hi:
            raise ValueError(
                f"Feature '{num_col}' ({num_val}) is outside dataset range [{lo}, {hi}]."
            )

        validated_num[num_col] = float(int(round(num_val))) if num_col == "Age" else float(num_val)

    encoded_cats: dict[str, int] = {
        cat_col: _encode_notebook_category(employee_details[cat_col], cat_col)
        for cat_col in CATEGORICAL_ENCODING_ORDER
    }

    # Resolve active backend model (defaults to employee_selary.pkl)
    requested_model_name = employee_details.get("Model_Name")
    if (
        artifact is not None
        and requested_model_name
        and "models" in artifact
        and requested_model_name in artifact["models"]
    ):
        active_model = artifact["models"][requested_model_name]
    elif artifact is not None and artifact.get("pipeline") is not None:
        active_model = artifact["pipeline"]
    else:
        active_model = load_backend_pickle_model(BACKEND_PKL_PATH)
        if active_model is None:
            raise FileNotFoundError(
                f"Backend model file not found at {BACKEND_PKL_PATH}."
            )

    default_tenure = min(
        int(round(validated_num["Experience_Years"])),
        int(aux_defaults.get("Company_Tenure", 5)),
    )
    full_row: dict[str, Any] = {
        "Age": float(validated_num["Age"]),
        "Gender": int(encoded_cats["Gender"]),
        "Education": int(encoded_cats["Education"]),
        "Experience_Years": float(validated_num["Experience_Years"]),
        "Department": int(encoded_cats["Department"]),
        "Job_Level": int(encoded_cats["Job_Level"]),
        "Performance_Rating": int(
            employee_details.get("Performance_Rating", aux_defaults.get("Performance_Rating", 3))
        ),
        "Certifications": int(
            employee_details.get("Certifications", aux_defaults.get("Certifications", 5))
        ),
        "Overtime_Hours": int(
            employee_details.get("Overtime_Hours", aux_defaults.get("Overtime_Hours", 30))
        ),
        "Remote_Work": int(encoded_cats["Remote_Work"]),
        "City": int(encoded_cats["City"]),
        "Company_Tenure": int(
            employee_details.get("Company_Tenure", default_tenure)
        ),
        "Projects_Completed": int(
            employee_details.get("Projects_Completed", aux_defaults.get("Projects_Completed", 15))
        ),
        "Skill_Score": float(validated_num["Skill_Score"]),
    }

    expected_cols = getattr(active_model, "feature_names_in_", None)
    if expected_cols is not None:
        ordered_cols = [str(c) for c in expected_cols]
    else:
        ordered_cols = NOTEBOOK_MODEL_COLUMNS

    if all(col in full_row for col in ordered_cols):
        input_df = pd.DataFrame([{col: full_row[col] for col in ordered_cols}], index=[0])
    else:
        raw_row = {col: employee_details[col] for col in FEATURE_COLUMNS}
        input_df = pd.DataFrame([raw_row], columns=FEATURE_COLUMNS)

    predicted_lpa = float(active_model.predict(input_df)[0])
    return round(max(0.0, predicted_lpa), 2)


# =============================================================================
# PURE-PYTHON CIRCULAR RADIAL DIALS & SALARY GAUGE VISUALIZATIONS
# =============================================================================


def build_circular_dial_figure(
    *,
    label: str,
    unit: str,
    value: float,
    min_val: float,
    max_val: float,
    is_int: bool,
    color_start: str,
    color_end: str,
) -> go.Figure:
    """Build a crisp, colour-themed circular radial dial using pure Python Plotly.

    Renders a smooth 270-degree circular progress arc, glowing circumferential handle,
    precision tick marks, centered numerical readout, and min/max bounds with zero
    external HTML/CSS/JS files.
    """
    span = max(1e-6, max_val - min_val)
    ratio = max(0.0, min(1.0, (value - min_val) / span))

    # 270-degree sweep clockwise from bottom-left (225 deg) to bottom-right (-45 deg)
    start_deg = 225.0
    sweep_deg = 270.0

    # Background track points
    bg_angles = np.linspace(start_deg, start_deg - sweep_deg, 96)
    bg_x = np.cos(np.radians(bg_angles))
    bg_y = np.sin(np.radians(bg_angles))

    # Active progress track points
    active_count = max(2, int(round(ratio * 96)))
    active_angles = np.linspace(start_deg, start_deg - ratio * sweep_deg, active_count)
    active_x = np.cos(np.radians(active_angles))
    active_y = np.sin(np.radians(active_angles))

    # Handle coordinates at active arc tip
    tip_angle = math.radians(start_deg - ratio * sweep_deg)
    tip_x = math.cos(tip_angle)
    tip_y = math.sin(tip_angle)

    # Outer calibration ticks
    tick_angles = np.linspace(start_deg, start_deg - sweep_deg, 29)
    tick_x_inactive: list[float | None] = []
    tick_y_inactive: list[float | None] = []
    tick_x_active: list[float | None] = []
    tick_y_active: list[float | None] = []

    for idx, deg in enumerate(tick_angles):
        rad = math.radians(float(deg))
        r_inner = 1.15
        r_outer = 1.24 if idx % 7 == 0 else 1.20
        t_ratio = idx / 28.0
        target_x = tick_x_active if t_ratio <= ratio + 1e-6 else tick_x_inactive
        target_y = tick_y_active if t_ratio <= ratio + 1e-6 else tick_y_inactive
        target_x.extend([r_inner * math.cos(rad), r_outer * math.cos(rad), None])
        target_y.extend([r_inner * math.sin(rad), r_outer * math.sin(rad), None])

    display_val = f"{int(round(value))}" if is_int else f"{value:.1f}"
    min_str = f"{int(round(min_val))}" if is_int else f"{min_val:.1f}"
    max_str = f"{int(round(max_val))}" if is_int else f"{max_val:.1f}"

    fig = go.Figure()

    # Recessed dark inner disc
    fig.add_shape(
        type="circle",
        x0=-0.82,
        y0=-0.82,
        x1=0.82,
        y1=0.82,
        fillcolor="#13192E",
        line=dict(color="rgba(255,255,255,0.08)", width=1),
        layer="below",
    )

    # Inactive ticks
    if tick_x_inactive:
        fig.add_trace(
            go.Scatter(
                x=tick_x_inactive,
                y=tick_y_inactive,
                mode="lines",
                line=dict(color="rgba(168,176,197,0.25)", width=1.4),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Active ticks
    if tick_x_active:
        fig.add_trace(
            go.Scatter(
                x=tick_x_active,
                y=tick_y_active,
                mode="lines",
                line=dict(color=color_end, width=1.8),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Background ring track
    fig.add_trace(
        go.Scatter(
            x=bg_x,
            y=bg_y,
            mode="lines",
            line=dict(color="rgba(168,176,197,0.16)", width=13),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Outer soft glow under active arc
    fig.add_trace(
        go.Scatter(
            x=active_x,
            y=active_y,
            mode="lines",
            line=dict(color=color_start, width=20),
            opacity=0.22,
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Active progress track (two-tone gradient approximation using mid-segment blend)
    mid_idx = max(1, len(active_x) // 2)
    fig.add_trace(
        go.Scatter(
            x=active_x[: mid_idx + 1],
            y=active_y[: mid_idx + 1],
            mode="lines",
            line=dict(color=color_start, width=12),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=active_x[mid_idx:],
            y=active_y[mid_idx:],
            mode="lines",
            line=dict(color=color_end, width=12),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Glowing handle halo + crisp circular knob
    fig.add_trace(
        go.Scatter(
            x=[tip_x],
            y=[tip_y],
            mode="markers",
            marker=dict(size=22, color=color_start, opacity=0.32),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[tip_x],
            y=[tip_y],
            mode="markers",
            marker=dict(
                size=14,
                color="#FFFFFF",
                line=dict(color=color_start, width=3.5),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        margin=dict(l=6, r=6, t=28, b=6),
        height=236,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=f"<b>{label}</b>",
            x=0.5,
            y=0.96,
            xanchor="center",
            font=dict(family="Outfit, sans-serif", size=13, color="#FFFFFF"),
        ),
        xaxis=dict(range=[-1.34, 1.34], visible=False, fixedrange=True),
        yaxis=dict(
            range=[-1.28, 1.34],
            visible=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        annotations=[
            dict(
                x=0,
                y=0.10,
                text=f"<b>{display_val}</b>",
                showarrow=False,
                font=dict(family="Outfit, sans-serif", size=34, color="#FFFFFF"),
            ),
            dict(
                x=0,
                y=-0.30,
                text=unit,
                showarrow=False,
                font=dict(family="Manrope, sans-serif", size=11, color="#A8B0C5"),
            ),
            dict(
                x=-0.78,
                y=-1.04,
                text=min_str,
                showarrow=False,
                font=dict(family="JetBrains Mono, monospace", size=11, color="#A8B0C5"),
            ),
            dict(
                x=0.78,
                y=-1.04,
                text=max_str,
                showarrow=False,
                font=dict(family="JetBrains Mono, monospace", size=11, color="#A8B0C5"),
            ),
        ],
    )
    return fig


def build_salary_gauge_figure(
    predicted_lpa: float,
    min_lpa: float,
    max_lpa: float,
) -> go.Figure:
    """Build the central emerald-to-blue-to-purple circular salary gauge in Plotly."""
    span = max(1.0, max_lpa - min_lpa)
    ratio = max(0.05, min(1.0, (predicted_lpa - min_lpa) / span))

    start_deg = 225.0
    sweep_deg = 270.0

    bg_angles = np.linspace(start_deg, start_deg - sweep_deg, 120)
    active_count = max(3, int(round(ratio * 120)))
    active_angles = np.linspace(start_deg, start_deg - ratio * sweep_deg, active_count)
    ax = np.cos(np.radians(active_angles))
    ay = np.sin(np.radians(active_angles))

    tip_rad = math.radians(start_deg - ratio * sweep_deg)
    tip_x, tip_y = math.cos(tip_rad), math.sin(tip_rad)

    seg1 = max(1, len(ax) // 3)
    seg2 = max(seg1 + 1, (2 * len(ax)) // 3)

    fig = go.Figure()

    # Dark inner dial disc
    fig.add_shape(
        type="circle",
        x0=-0.84,
        y0=-0.84,
        x1=0.84,
        y1=0.84,
        fillcolor="#101426",
        line=dict(color="rgba(255,255,255,0.09)", width=1),
        layer="below",
    )

    # Background ring
    fig.add_trace(
        go.Scatter(
            x=np.cos(np.radians(bg_angles)),
            y=np.sin(np.radians(bg_angles)),
            mode="lines",
            line=dict(color="rgba(168,176,197,0.15)", width=16),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Soft glow underlay
    fig.add_trace(
        go.Scatter(
            x=ax,
            y=ay,
            mode="lines",
            line=dict(color="#4285FF", width=26),
            opacity=0.20,
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Emerald -> Electric Blue -> Purple active arc
    fig.add_trace(
        go.Scatter(
            x=ax[: seg1 + 1],
            y=ay[: seg1 + 1],
            mode="lines",
            line=dict(color="#19C991", width=15),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ax[seg1 : seg2 + 1],
            y=ay[seg1 : seg2 + 1],
            mode="lines",
            line=dict(color="#4285FF", width=15),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ax[seg2:],
            y=ay[seg2:],
            mode="lines",
            line=dict(color="#A855F7", width=15),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Glowing tip indicator
    fig.add_trace(
        go.Scatter(
            x=[tip_x],
            y=[tip_y],
            mode="markers",
            marker=dict(size=24, color="#A855F7", opacity=0.34),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[tip_x],
            y=[tip_y],
            mode="markers",
            marker=dict(size=13, color="#FFFFFF", line=dict(color="#A855F7", width=3)),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=286,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-1.28, 1.28], visible=False, fixedrange=True),
        yaxis=dict(
            range=[-1.24, 1.28],
            visible=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        annotations=[
            dict(
                x=0,
                y=0.10,
                text=f"<b>₹{predicted_lpa:.2f} LPA</b>",
                showarrow=False,
                font=dict(family="Outfit, sans-serif", size=27, color="#FFFFFF"),
            ),
            dict(
                x=0,
                y=-0.26,
                text="PREDICTED ANNUAL SALARY",
                showarrow=False,
                font=dict(family="Manrope, sans-serif", size=10, color="#A8B0C5"),
            ),
            dict(
                x=-0.76,
                y=-1.02,
                text=f"₹{min_lpa:.0f}L",
                showarrow=False,
                font=dict(family="JetBrains Mono, monospace", size=11, color="#A8B0C5"),
            ),
            dict(
                x=0.0,
                y=-1.02,
                text="Dataset Scale",
                showarrow=False,
                font=dict(family="Manrope, sans-serif", size=10, color="#A8B0C5"),
            ),
            dict(
                x=0.76,
                y=-1.02,
                text=f"₹{max_lpa:.0f}L",
                showarrow=False,
                font=dict(family="JetBrains Mono, monospace", size=11, color="#A8B0C5"),
            ),
        ],
    )
    return fig


def build_report_csv(
    details: dict[str, Any],
    predicted_lpa: float,
    avg_lpa: float,
    diff_lpa: float,
    selected_model_name: str,
    timestamp_iso: str,
) -> bytes:
    """Create downloadable CSV report bytes for the current salary prediction."""
    row: dict[str, Any] = {
        "Age": details["Age"],
        "Gender": details["Gender"],
        "Education": details["Education"],
        "Experience_Years": details["Experience_Years"],
        "Department": details["Department"],
        "Job_Level": details["Job_Level"],
        "Performance_Rating": details.get("Performance_Rating", NOTEBOOK_CELL82_DEFAULTS["Performance_Rating"]),
        "Certifications": details.get("Certifications", NOTEBOOK_CELL82_DEFAULTS["Certifications"]),
        "Overtime_Hours": details.get("Overtime_Hours", NOTEBOOK_CELL82_DEFAULTS["Overtime_Hours"]),
        "Remote_Work": details["Remote_Work"],
        "City": details["City"],
        "Company_Tenure": details.get("Company_Tenure", NOTEBOOK_CELL82_DEFAULTS["Company_Tenure"]),
        "Projects_Completed": details.get("Projects_Completed", NOTEBOOK_CELL82_DEFAULTS["Projects_Completed"]),
        "Skill_Score": details["Skill_Score"],
        "Predicted_Salary_LPA": f"{predicted_lpa:.2f}",
        "Dataset_Average_Salary_LPA": f"{avg_lpa:.2f}",
        "Difference_From_Average_LPA": f"{diff_lpa:.2f}",
        "Selected_Model": selected_model_name,
        "Prediction_Timestamp": timestamp_iso,
    }
    report_df = pd.DataFrame([row])
    return report_df.to_csv(index=False).encode("utf-8")


# =============================================================================
# INLINE STUDIO THEME STYLING (NO EXTERNAL CSS FILE)
# =============================================================================

STUDIO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;600&family=Manrope:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

:root {
  --bg-primary: #101426;
  --bg-secondary: #171D32;
  --bg-card: #1D2540;
  --text-primary: #FFFFFF;
  --text-secondary: #A8B0C5;
  --emerald: #19C991;
  --blue: #4285FF;
  --purple: #A855F7;
  --pink: #EC4899;
  --orange: #FF9D45;
  --cyan: #22D3EE;
}

#MainMenu, footer, header[data-testid="stHeader"], section[data-testid="stSidebar"] {
  display: none !important;
}

.stApp {
  background-color: var(--bg-primary) !important;
  background-image:
    radial-gradient(ellipse 68% 42% at 15% -5%, rgba(66, 133, 255, 0.14) 0%, transparent 65%),
    radial-gradient(ellipse 58% 38% at 85% 4%, rgba(168, 85, 247, 0.11) 0%, transparent 65%),
    radial-gradient(ellipse 55% 40% at 50% 100%, rgba(25, 201, 145, 0.08) 0%, transparent 65%) !important;
  color: var(--text-primary) !important;
  font-family: "Manrope", system-ui, sans-serif !important;
}

.block-container {
  max-width: 1120px !important;
  padding-top: 1.25rem !important;
  padding-bottom: 3.5rem !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
}

.studio-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding-bottom: 1.15rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  margin-bottom: 1.75rem;
}
.brand-lockup { display: flex; align-items: center; gap: 0.85rem; }
.brand-logo {
  width: 42px;
  height: 42px;
  border-radius: 12px;
  background: var(--bg-card);
  border: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.brand-title {
  font-family: "Outfit", system-ui, sans-serif;
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  color: #FFFFFF;
}
.brand-sub { font-size: 0.78rem; color: var(--text-secondary); margin-top: 2px; }
.status-strip {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.6rem;
  font-size: 0.78rem;
  color: var(--text-secondary);
}
.status-sep { color: rgba(255, 255, 255, 0.22); }
.st-ok-emerald { color: #19C991; font-weight: 600; }
.st-ok-blue { color: #4285FF; font-weight: 600; }
.st-ok-cyan { color: #22D3EE; font-weight: 600; }
.st-warn { color: #FF9D45; font-weight: 600; }

.studio-hero {
  text-align: center;
  max-width: 640px;
  margin: 0 auto 2rem auto;
}
.studio-hero h1 {
  font-family: "Outfit", system-ui, sans-serif;
  font-size: clamp(1.9rem, 3.6vw, 2.5rem);
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #FFFFFF;
  margin-bottom: 0.5rem;
}
.studio-hero p {
  font-size: 0.96rem;
  color: var(--text-secondary);
  line-height: 1.6;
}

.studio-panel-header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: flex-end;
  gap: 0.5rem;
  margin-bottom: 1.25rem;
}
.studio-section-title {
  font-family: "Outfit", system-ui, sans-serif;
  font-size: 1.35rem;
  font-weight: 700;
  color: #FFFFFF;
  margin: 0;
}
.studio-section-desc {
  font-size: 0.88rem;
  color: var(--text-secondary);
  margin: 0.2rem 0 0 0;
}
.studio-model-meta {
  font-size: 0.78rem;
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}
.studio-divider {
  border: 0;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  margin: 1.5rem 0 1.35rem 0;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(23, 29, 50, 0.92) !important;
  border: 1px solid rgba(255, 255, 255, 0.09) !important;
  border-radius: 22px !important;
  padding: 1.4rem 1.6rem !important;
  box-shadow: 0 24px 60px -28px rgba(0, 0, 0, 0.65);
}

label, .stSelectbox label, .stRadio label, .stNumberInput label, .stSlider label {
  color: var(--text-secondary) !important;
  font-size: 0.84rem !important;
  font-weight: 500 !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
  background-color: #101426 !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  border-radius: 12px !important;
  color: #FFFFFF !important;
}

div.stButton > button[kind="primary"] {
  background: linear-gradient(100deg, #19C991 0%, #24B8B8 46%, #4285FF 100%) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 16px !important;
  height: 3.45rem !important;
  font-family: "Outfit", system-ui, sans-serif !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  box-shadow: 0 10px 30px -8px rgba(34, 184, 184, 0.45) !important;
}
div.stButton > button[kind="primary"]:hover {
  filter: brightness(1.06);
}

div.stButton > button[kind="secondary"],
div.stDownloadButton > button {
  background-color: #1D2540 !important;
  color: #FFFFFF !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.9rem;
  margin-bottom: 1.25rem;
}
@media (max-width: 760px) {
  .metric-grid { grid-template-columns: 1fr; }
}
.metric-card {
  background: rgba(29, 37, 64, 0.88);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  padding: 1rem 1.1rem;
}
.metric-label { font-size: 0.76rem; color: var(--text-secondary); }
.metric-value {
  font-family: "Outfit", system-ui, sans-serif;
  font-size: 1.32rem;
  font-weight: 700;
  color: #FFFFFF;
  margin-top: 0.35rem;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.metric-value.up { color: #19C991; }
.metric-value.down { color: #FF9D45; }
.metric-sub { font-size: 0.74rem; color: var(--text-secondary); margin-top: 0.3rem; }

.profile-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.65rem 1.4rem;
  padding-top: 0.85rem;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}
@media (max-width: 640px) {
  .profile-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.profile-item {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  padding-bottom: 0.4rem;
  font-size: 0.8rem;
}
.profile-k { color: var(--text-secondary); }
.profile-v { color: #FFFFFF; font-weight: 600; font-variant-numeric: tabular-nums; }

.eval-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.84rem;
  margin-top: 0.6rem;
}
.eval-table th {
  text-align: left;
  color: var(--text-secondary);
  font-weight: 600;
  padding: 0.55rem 0.6rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.eval-table td {
  padding: 0.55rem 0.6rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  color: #FFFFFF;
}
.eval-table td.num, .eval-table th.num {
  text-align: right;
  font-family: "JetBrains Mono", monospace;
  font-variant-numeric: tabular-nums;
}
.eval-table tr.selected-row { background: rgba(66, 133, 255, 0.1); }
</style>
"""


# =============================================================================
# SESSION STATE & CACHING HELPERS
# =============================================================================


def init_input_state(bounds: dict[str, Any], force: bool = False) -> None:
    if force:
        st.session_state["reset_nonce"] = st.session_state.get("reset_nonce", 0) + 1
        st.session_state["prediction_result"] = None

    nonce = st.session_state.get("reset_nonce", 0)
    age_default = int(
        np.clip(
            NOTEBOOK_CELL82_DEFAULTS["Age"],
            bounds["Age"]["min"],
            bounds["Age"]["max"],
        )
    )
    exp_default = float(
        np.clip(
            NOTEBOOK_CELL82_DEFAULTS["Experience_Years"],
            bounds["Experience_Years"]["min"],
            bounds["Experience_Years"]["max"],
        )
    )
    skill_default = float(
        np.clip(
            NOTEBOOK_CELL82_DEFAULTS["Skill_Score"],
            bounds["Skill_Score"]["min"],
            bounds["Skill_Score"]["max"],
        )
    )
    defaults = {
        f"age_val_{nonce}": age_default,
        f"exp_val_{nonce}": exp_default,
        f"skill_val_{nonce}": skill_default,
        f"gender_val_{nonce}": NOTEBOOK_CELL82_DEFAULTS["Gender"],
        f"education_val_{nonce}": NOTEBOOK_CELL82_DEFAULTS["Education"],
        f"department_val_{nonce}": NOTEBOOK_CELL82_DEFAULTS["Department"],
        f"job_level_val_{nonce}": NOTEBOOK_CELL82_DEFAULTS["Job_Level"],
        f"remote_val_{nonce}": NOTEBOOK_CELL82_DEFAULTS["Remote_Work"],
        f"city_val_{nonce}": NOTEBOOK_CELL82_DEFAULTS["City"],
        f"perf_val_{nonce}": int(NOTEBOOK_CELL82_DEFAULTS["Performance_Rating"]),
        f"cert_val_{nonce}": int(NOTEBOOK_CELL82_DEFAULTS["Certifications"]),
        f"overtime_val_{nonce}": int(NOTEBOOK_CELL82_DEFAULTS["Overtime_Hours"]),
        f"tenure_val_{nonce}": int(NOTEBOOK_CELL82_DEFAULTS["Company_Tenure"]),
        f"projects_val_{nonce}": int(NOTEBOOK_CELL82_DEFAULTS["Projects_Completed"]),
    }
    for k, v in defaults.items():
        if force or k not in st.session_state:
            st.session_state[k] = v


@st.cache_resource(show_spinner=False)
def get_cached_pipeline(csv_bytes: bytes, source_name: str) -> dict[str, Any]:
    return load_or_train_artifact(csv_bytes, source_name)


# -----------------------------------------------------------------------------
# DRAGGABLE CIRCULAR DIAL (inline Streamlit Components v2 — no external files)
# -----------------------------------------------------------------------------
# The dial is rendered and animated entirely in the browser, so dragging is
# 60 fps smooth. Python is only notified once, when the drag is released (or
# after a short pause for mouse-wheel / keyboard input), which avoids reruns
# while the knob is moving.

RADIAL_DIAL_CSS = """
:host { display: block; }
.rd-root {
  position: relative;
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  user-select: none;
  -webkit-user-select: none;
  font-family: "Outfit", "Manrope", system-ui, sans-serif;
}
.rd-head {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 0.15rem 0.1rem 0.15rem;
  box-sizing: border-box;
}
.rd-label { color: #FFFFFF; font-size: 0.86rem; font-weight: 700; letter-spacing: 0.06em; }
.rd-hint {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.64rem;
  letter-spacing: 0.08em;
  color: #A8B0C5;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 999px;
  padding: 0.12rem 0.5rem;
}
.rd-svg {
  width: 100%;
  max-width: 270px;
  height: auto;
  display: block;
  touch-action: none;
  cursor: grab;
  outline: none;
  overflow: visible;
}
.rd-root.dragging .rd-svg { cursor: grabbing; }
.rd-svg:focus-visible .rd-bezel { stroke: rgba(255, 255, 255, 0.35); }
.rd-bezel { fill: none; stroke: rgba(255, 255, 255, 0.07); stroke-width: 1.2; }
.rd-track { fill: none; stroke: rgba(168, 176, 197, 0.14); stroke-width: 16; stroke-linecap: round; }
.rd-prog { fill: none; stroke-width: 16; stroke-linecap: round; }
.rd-face { fill: #13192E; stroke: rgba(255, 255, 255, 0.08); stroke-width: 1; }
.rd-tick { stroke: rgba(168, 176, 197, 0.22); stroke-width: 1.4; stroke-linecap: round; transition: stroke 0.15s ease; }
.rd-tick.major { stroke-width: 2.2; }
.rd-val { fill: #FFFFFF; font-size: 46px; font-weight: 800; text-anchor: middle; letter-spacing: -1px; }
.rd-unit { fill: #A8B0C5; font-size: 11.5px; font-weight: 600; text-anchor: middle; letter-spacing: 1.6px; }
.rd-end { fill: #A8B0C5; font-family: "JetBrains Mono", monospace; font-size: 11px; text-anchor: middle; }
.rd-knob { fill: #FFFFFF; stroke-width: 4; transition: r 0.15s ease; }
.rd-knob-glow { opacity: 0.35; transition: r 0.15s ease, opacity 0.15s ease; }
.rd-root.dragging .rd-knob-glow { opacity: 0.55; }
"""

RADIAL_DIAL_JS = r"""
const CX = 130, CY = 130, R = 96;
const START = 120, SWEEP = 300;            // 300-degree circular sweep, gap at the bottom
const CIRC = 2 * Math.PI * R;
const ARC = CIRC * SWEEP / 360;
const TICKS = 60;

function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
function polar(deg, r) {
  const a = deg * Math.PI / 180;
  return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
}
function ratioOf(cfg, v) {
  const span = (cfg.max - cfg.min) || 1;
  return clamp((v - cfg.min) / span, 0, 1);
}
function snapValue(cfg, v) {
  const step = cfg.step || 1;
  const snapped = Math.round((v - cfg.min) / step) * step + cfg.min;
  return Number(clamp(snapped, cfg.min, cfg.max).toFixed(cfg.decimals || 0));
}
function valueAt(cfg, ratio) { return snapValue(cfg, cfg.min + ratio * (cfg.max - cfg.min)); }
function fmt(cfg, v) { return Number(v).toFixed(cfg.decimals || 0); }

function createDial(parent, cfg) {
  const uid = cfg.uid || ("d" + Math.random().toString(36).slice(2));
  let ticks = "";
  for (let i = 0; i <= TICKS; i++) {
    const deg = START + SWEEP * i / TICKS;
    const major = i % 10 === 0;
    const [x1, y1] = polar(deg, R + 15);
    const [x2, y2] = polar(deg, R + (major ? 25 : 20));
    ticks += `<line class="rd-tick${major ? " major" : ""}" x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}"/>`;
  }
  const [minX, minY] = polar(START, R);
  const [maxX, maxY] = polar(START + SWEEP, R);

  const root = document.createElement("div");
  root.className = "rd-root";
  root.innerHTML = `
    <div class="rd-head"><span class="rd-label"></span><span class="rd-hint">DRAG ⟳</span></div>
    <svg class="rd-svg" viewBox="0 0 260 262" role="slider" tabindex="0">
      <defs>
        <linearGradient id="g-${uid}" x1="0" y1="1" x2="1" y2="0">
          <stop offset="0%" class="rd-s1"/><stop offset="100%" class="rd-s2"/>
        </linearGradient>
        <filter id="f-${uid}" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <radialGradient id="h-${uid}" cx="50%" cy="50%" r="50%">
          <stop offset="0%" class="rd-h1" stop-opacity="0.22"/>
          <stop offset="100%" stop-color="#101426" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <circle cx="${CX}" cy="${CY}" r="128" fill="url(#h-${uid})"/>
      <circle class="rd-bezel" cx="${CX}" cy="${CY}" r="${R + 31}"/>
      <g class="rd-ticks">${ticks}</g>
      <circle class="rd-track" cx="${CX}" cy="${CY}" r="${R}"
              stroke-dasharray="${ARC} ${CIRC}" transform="rotate(${START} ${CX} ${CY})"/>
      <circle class="rd-prog" cx="${CX}" cy="${CY}" r="${R}" stroke="url(#g-${uid})"
              filter="url(#f-${uid})" stroke-dasharray="0 ${CIRC}" transform="rotate(${START} ${CX} ${CY})"/>
      <circle class="rd-face" cx="${CX}" cy="${CY}" r="72"/>
      <text class="rd-val" x="${CX}" y="${CY + 10}">0</text>
      <text class="rd-unit" x="${CX}" y="${CY + 34}"></text>
      <text class="rd-end rd-min" x="${minX.toFixed(1)}" y="${(minY + 36).toFixed(1)}"></text>
      <text class="rd-end rd-max" x="${maxX.toFixed(1)}" y="${(maxY + 36).toFixed(1)}"></text>
      <circle class="rd-knob-glow" r="16" cx="0" cy="0"/>
      <circle class="rd-knob" r="9" cx="0" cy="0"/>
    </svg>`;
  parent.appendChild(root);

  const svg = root.querySelector(".rd-svg");
  const s = {
    root, svg, cfg,
    prog: root.querySelector(".rd-prog"),
    knob: root.querySelector(".rd-knob"),
    glow: root.querySelector(".rd-knob-glow"),
    valText: root.querySelector(".rd-val"),
    tickEls: Array.from(root.querySelectorAll(".rd-tick")),
    ratio: 0, target: 0, raf: null,
    dragging: false, pending: null,
    committed: null, lastExternal: undefined, send: null,
  };

  s.applyTheme = () => {
    const c = s.cfg;
    root.querySelector(".rd-s1").setAttribute("stop-color", c.c1);
    root.querySelector(".rd-s2").setAttribute("stop-color", c.c2);
    root.querySelector(".rd-h1").setAttribute("stop-color", c.c1);
    root.querySelector(".rd-label").textContent = c.label;
    root.querySelector(".rd-unit").textContent = (c.unit || "").toUpperCase();
    root.querySelector(".rd-min").textContent = fmt(c, c.min);
    root.querySelector(".rd-max").textContent = fmt(c, c.max);
    s.knob.setAttribute("stroke", c.c2);
    s.glow.setAttribute("fill", c.c2);
    svg.setAttribute("aria-label", c.label);
    svg.setAttribute("aria-valuemin", c.min);
    svg.setAttribute("aria-valuemax", c.max);
  };

  s.draw = () => {
    const c = s.cfg;
    const r = s.ratio;
    s.prog.setAttribute("stroke-dasharray", `${(ARC * r).toFixed(2)} ${CIRC}`);
    const [kx, ky] = polar(START + SWEEP * r, R);
    s.knob.setAttribute("cx", kx.toFixed(2)); s.knob.setAttribute("cy", ky.toFixed(2));
    s.glow.setAttribute("cx", kx.toFixed(2)); s.glow.setAttribute("cy", ky.toFixed(2));
    const shown = s.dragging || s.pending ? valueAt(c, s.target) : valueAt(c, r);
    s.valText.textContent = fmt(c, shown);
    svg.setAttribute("aria-valuenow", shown);
    const lit = Math.round(r * TICKS);
    for (let i = 0; i < s.tickEls.length; i++) {
      s.tickEls[i].style.stroke = i <= lit ? c.c2 : "";
    }
  };

  // Critically-damped easing toward the target angle => buttery-smooth motion.
  s.loop = () => {
    const diff = s.target - s.ratio;
    if (Math.abs(diff) < 0.0004) {
      s.ratio = s.target; s.draw(); s.raf = null; return;
    }
    s.ratio += diff * (s.dragging ? 0.38 : 0.16);
    s.draw();
    s.raf = requestAnimationFrame(s.loop);
  };
  s.animateTo = (r) => { s.target = clamp(r, 0, 1); if (!s.raf) s.raf = requestAnimationFrame(s.loop); };

  s.commit = (v) => {
    if (v === s.committed) return;
    s.committed = v;
    if (s.send) s.send("value", { v: v, seq: Date.now() });
  };

  const toSvg = (e) => {
    const pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const m = svg.getScreenCTM();
    return m ? pt.matrixTransform(m.inverse()) : { x: CX, y: CY };
  };

  const ratioFromPoint = (p, allowJump) => {
    const ang = Math.atan2(p.y - CY, p.x - CX) * 180 / Math.PI;
    let rel = ((ang - START) % 360 + 360) % 360;
    if (rel > SWEEP) rel = rel > SWEEP + (360 - SWEEP) / 2 ? 0 : SWEEP;  // inside the bottom gap
    let r = rel / SWEEP;
    // Prevent wrapping straight from max to min (or vice versa) across the gap.
    if (!allowJump && Math.abs(r - s.target) > 0.5) r = s.target > 0.5 ? 1 : 0;
    return r;
  };

  svg.addEventListener("pointerdown", (e) => {
    const p = toSvg(e);
    if (Math.hypot(p.x - CX, p.y - CY) < 28) return;
    e.preventDefault();
    svg.setPointerCapture(e.pointerId);
    svg.focus({ preventScroll: true });
    s.dragging = true;
    root.classList.add("dragging");
    s.animateTo(ratioFromPoint(p, true));
  });
  svg.addEventListener("pointermove", (e) => {
    if (!s.dragging) return;
    e.preventDefault();
    s.animateTo(ratioFromPoint(toSvg(e), false));
  });
  const endDrag = () => {
    if (!s.dragging) return;
    s.dragging = false;
    root.classList.remove("dragging");
    const v = valueAt(s.cfg, s.target);
    s.animateTo(ratioOf(s.cfg, v));
    s.commit(v);
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);
  svg.addEventListener("lostpointercapture", endDrag);

  const stepBy = (n) => {
    const c = s.cfg;
    const v = snapValue(c, valueAt(c, s.target) + n * (c.step || 1));
    s.animateTo(ratioOf(c, v));
    clearTimeout(s.pending);
    s.pending = setTimeout(() => { s.pending = null; s.commit(v); s.draw(); }, 380);
  };
  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    stepBy(e.deltaY < 0 ? 1 : -1);
  }, { passive: false });
  svg.addEventListener("keydown", (e) => {
    const big = Math.max(1, Math.round(((s.cfg.max - s.cfg.min) / (s.cfg.step || 1)) / 10));
    const map = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1, PageUp: big, PageDown: -big };
    if (e.key in map) { e.preventDefault(); stepBy(map[e.key]); }
    else if (e.key === "Home") { e.preventDefault(); stepBy(-1e9); }
    else if (e.key === "End") { e.preventDefault(); stepBy(1e9); }
  });

  return s;
}

export default function (component) {
  const { data, setStateValue, parentElement } = component;
  const cfg = data || {};
  let s = parentElement.__radialDial;
  if (!s) {
    s = createDial(parentElement, cfg);
    parentElement.__radialDial = s;
  }
  s.cfg = cfg;
  s.send = setStateValue;
  s.applyTheme();
  // Accept value changes coming from Python (reset, +/- buttons) unless the
  // user is currently interacting with the dial.
  if (!s.dragging && !s.pending && cfg.value !== s.lastExternal) {
    s.lastExternal = cfg.value;
    s.committed = cfg.value;
    s.animateTo(ratioOf(cfg, cfg.value));
  } else {
    s.draw();
  }
}
"""


@st.cache_resource(show_spinner=False)
def get_radial_dial_component() -> Any:
    """Register the inline draggable dial component once per server process."""
    import streamlit.components.v2 as components_v2

    return components_v2.component(
        "salary_studio_radial_dial",
        css=RADIAL_DIAL_CSS,
        js=RADIAL_DIAL_JS,
    )


def _sync_dial_value(
    comp_key: str,
    state_key: str,
    min_val: float,
    max_val: float,
    step: float,
    is_int: bool,
) -> None:
    """Callback: copy the value released on the circular dial into session state."""
    comp_state = st.session_state.get(comp_key)
    payload = comp_state.get("value") if isinstance(comp_state, dict) else None
    raw = payload.get("v") if isinstance(payload, dict) else payload
    try:
        num = float(raw)
    except (TypeError, ValueError):
        return
    if not math.isfinite(num):
        return
    num = min(max(num, float(min_val)), float(max_val))
    num = round(round((num - min_val) / step) * step + min_val, 4)
    st.session_state[state_key] = int(round(num)) if is_int else float(num)


def _step_dial_value(
    state_key: str, delta: float, min_val: float, max_val: float, is_int: bool
) -> None:
    """Callback for the fine +/- buttons."""
    nxt = min(max(float(st.session_state[state_key]) + delta, float(min_val)), float(max_val))
    st.session_state[state_key] = int(round(nxt)) if is_int else round(nxt, 2)


def render_dial_card(
    *,
    label: str,
    unit: str,
    state_key: str,
    min_val: float,
    max_val: float,
    step: float,
    is_int: bool,
    color_start: str,
    color_end: str,
) -> float:
    """Render a draggable circular dial (drag the knob around the ring, scroll, or use arrow keys)."""
    comp_key = f"radial_{state_key}"
    step = float(step) if step else 1.0
    decimals = 0 if (is_int or step.is_integer()) else 1

    with st.container(border=True):
        dial = get_radial_dial_component()
        dial(
            key=comp_key,
            data={
                "uid": comp_key,
                "label": label,
                "unit": unit,
                "value": float(st.session_state[state_key]),
                "min": float(min_val),
                "max": float(max_val),
                "step": step,
                "decimals": decimals,
                "c1": color_start,
                "c2": color_end,
            },
            on_value_change=lambda: _sync_dial_value(
                comp_key, state_key, min_val, max_val, step, is_int
            ),
        )

        # Fine-tune buttons (also useful on devices without precise pointer input)
        b1, b2 = st.columns(2, gap="small")
        with b1:
            st.button(
                "−",
                key=f"dec_{state_key}",
                width="stretch",
                on_click=_step_dial_value,
                args=(state_key, -step, min_val, max_val, is_int),
            )
        with b2:
            st.button(
                "+",
                key=f"inc_{state_key}",
                width="stretch",
                on_click=_step_dial_value,
                args=(state_key, step, min_val, max_val, is_int),
            )

    return int(st.session_state[state_key]) if is_int else float(st.session_state[state_key])


# =============================================================================
# MAIN SINGLE-SCREEN APPLICATION
# =============================================================================


def render_header(
    dataset_loaded: bool,
    ml_ready: bool,
    row_count: int | None,
    pkl_connected: bool = False,
) -> None:
    ds_text = (
        f"<span class='st-ok-emerald'>Dataset Loaded</span> ({row_count:,})"
        if (dataset_loaded and row_count is not None)
        else "<span class='st-warn'>Dataset Required</span>"
    )
    ml_text = (
        "<span class='st-ok-blue'>Backend: employee_selary.pkl Connected</span>"
        if pkl_connected
        else (
            "<span class='st-ok-blue'>ML Engine Ready</span>"
            if ml_ready
            else "<span class='st-warn'>ML Engine Standby</span>"
        )
    )
    sys_text = (
        "<span class='st-ok-cyan'>Prediction System Active</span>"
        if (dataset_loaded and ml_ready)
        else "<span class='st-warn'>Awaiting Dataset</span>"
    )

    st.markdown(
        f"""
        <div class="studio-header">
          <div class="brand-lockup">
            <div class="brand-logo" aria-hidden="true">
              <svg width="24" height="24" viewBox="0 0 36 36" fill="none">
                <defs>
                  <linearGradient id="studioLogoGrad" x1="0" y1="36" x2="36" y2="0" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stop-color="#19C991" />
                    <stop offset="55%" stop-color="#4285FF" />
                    <stop offset="100%" stop-color="#A855F7" />
                  </linearGradient>
                </defs>
                <circle cx="18" cy="18" r="13" stroke="url(#studioLogoGrad)" stroke-width="3.5"
                        stroke-linecap="round" stroke-dasharray="62 22" transform="rotate(135 18 18)" />
                <circle cx="18" cy="18" r="4.5" fill="#FFFFFF" />
              </svg>
            </div>
            <div>
              <div class="brand-title">SALARY PREDICTION STUDIO</div>
              <div class="brand-sub">Machine Learning Compensation Engine</div>
            </div>
          </div>
          <div class="status-strip">
            <span>{ds_text}</span>
            <span class="status-sep">·</span>
            <span>{ml_text}</span>
            <span class="status-sep">·</span>
            <span>{sys_text}</span>
          </div>
        </div>
        <div class="studio-hero">
          <h1>Discover Your Estimated Salary</h1>
          <p>Adjust employee characteristics and let our machine learning model estimate annual compensation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Salary Prediction Studio",
        page_icon="💠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(STUDIO_CSS, unsafe_allow_html=True)

    # Load dataset from disk or in-page CSV uploader (never fabricates fake employee data)
    csv_bytes: bytes | None = st.session_state.get("uploaded_csv_bytes")
    source_name: str = st.session_state.get("uploaded_csv_name", DEFAULT_DATASET_PATH.name)

    if csv_bytes is None and DEFAULT_DATASET_PATH.exists():
        try:
            csv_bytes = DEFAULT_DATASET_PATH.read_bytes()
            source_name = DEFAULT_DATASET_PATH.name
        except Exception as exc:
            st.error(f"Could not read {DEFAULT_DATASET_PATH.name}: {exc}")

    artifact: dict[str, Any] | None = None
    load_error: str | None = None

    if csv_bytes is not None:
        try:
            with st.spinner("Loading dataset and training regression models..."):
                artifact = get_cached_pipeline(csv_bytes, source_name)
        except Exception as exc:
            load_error = str(exc)

    dataset_loaded = artifact is not None
    ml_ready = artifact is not None and artifact.get("pipeline") is not None
    pkl_connected = bool(artifact is not None and artifact.get("pkl_connected"))
    row_count = artifact["dataset_meta"]["row_count"] if artifact is not None else None

    # SECTION A — TOP HEADER
    render_header(dataset_loaded, ml_ready, row_count, pkl_connected=pkl_connected)

    # Single-page CSV uploader if employee_salary_dataset.csv is missing or invalid
    if artifact is None:
        with st.container(border=True):
            st.markdown(
                """
                <h2 class="studio-section-title">Load Employee Salary Dataset</h2>
                <p class="studio-section-desc">
                  Place <code>employee_salary_dataset.csv</code> next to <code>app.py</code>,
                  or upload your CSV file below to train all four regression models.
                </p>
                """,
                unsafe_allow_html=True,
            )
            if load_error:
                st.error(f"Dataset validation error: {load_error}")

            uploaded = st.file_uploader(
                "Upload employee_salary_dataset.csv",
                type=["csv"],
                help=f"Required columns: {', '.join(REQUIRED_COLUMNS)}",
            )
            if uploaded is not None:
                st.session_state["uploaded_csv_bytes"] = uploaded.getvalue()
                st.session_state["uploaded_csv_name"] = uploaded.name
                st.session_state["prediction_result"] = None
                st.rerun()
        return

    meta = artifact["dataset_meta"]
    bounds = meta["bounds"]
    selected_metrics = artifact["selected_metrics"]

    if st.session_state.get("active_dataset_hash") != artifact["dataset_hash"]:
        st.session_state["active_dataset_hash"] = artifact["dataset_hash"]
        init_input_state(bounds, force=True)
    else:
        init_input_state(bounds, force=False)

    nonce = st.session_state.get("reset_nonce", 0)
    age_key = f"age_val_{nonce}"
    exp_key = f"exp_val_{nonce}"
    skill_key = f"skill_val_{nonce}"

    # SECTION B — EMPLOYEE PROFILE CONFIGURATION
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="studio-panel-header">
              <div>
                <h2 class="studio-section-title">Employee Profile Configuration</h2>
                <p class="studio-section-desc">
                  Configure the employee parameters to generate a personalized salary prediction.
                </p>
              </div>
              <div class="studio-model-meta">
                Active model: <strong style="color:#FFFFFF">{selected_metrics['model_name']}</strong>
                &nbsp;·&nbsp; R² <span style="color:#19C991;font-family:'JetBrains Mono',monospace">{selected_metrics['r2_score']:.4f}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # THREE INTERACTIVE CIRCULAR RADIAL DIALS
        d1, d2, d3 = st.columns(3, gap="medium")
        with d1:
            age_val = render_dial_card(
                label="EMPLOYEE AGE",
                unit="Years Old",
                state_key=age_key,
                min_val=bounds["Age"]["min"],
                max_val=bounds["Age"]["max"],
                step=1,
                is_int=True,
                color_start="#4285FF",
                color_end="#22D3EE",
            )
        with d2:
            exp_val = render_dial_card(
                label="PROFESSIONAL EXPERIENCE",
                unit="Years of Experience",
                state_key=exp_key,
                min_val=bounds["Experience_Years"]["min"],
                max_val=bounds["Experience_Years"]["max"],
                step=bounds["Experience_Years"]["step"],
                is_int=bounds["Experience_Years"]["is_int"],
                color_start="#A855F7",
                color_end="#EC4899",
            )
        with d3:
            skill_val = render_dial_card(
                label="SKILL SCORE",
                unit="Skill Rating",
                state_key=skill_key,
                min_val=bounds["Skill_Score"]["min"],
                max_val=bounds["Skill_Score"]["max"],
                step=bounds["Skill_Score"]["step"],
                is_int=bounds["Skill_Score"]["is_int"],
                color_start="#19C991",
                color_end="#22D3EE",
            )

        st.markdown('<hr class="studio-divider" />', unsafe_allow_html=True)

        # EMPLOYEE DETAILS — EXACT CATEGORICAL FIELDS
        st.markdown(
            """
            <div style="margin-bottom: 1rem;">
              <h3 style="font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:700;color:#FFFFFF;margin:0;">
                Employee Details
              </h3>
              <p style="font-size:0.8rem;color:#A8B0C5;margin:0.15rem 0 0 0;">
                Exact categorical features aligned with the original trained dataset encoding
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        r1_c1, r1_c2, r1_c3 = st.columns(3, gap="medium")
        with r1_c1:
            gender_val = st.selectbox(
                "Gender (Male=0, Female=1)",
                options=UI_CATEGORY_OPTIONS["Gender"],
                key=f"gender_val_{nonce}",
            )
        with r1_c2:
            edu_val = st.selectbox(
                "Education (Diploma=0, Bachelor=1, Master=2, PhD=3)",
                options=UI_CATEGORY_OPTIONS["Education"],
                key=f"education_val_{nonce}",
            )
        with r1_c3:
            dept_val = st.selectbox(
                "Department (Operations=0, Marketing=1, Sales=2, IT=3, HR=4)",
                options=UI_CATEGORY_OPTIONS["Department"],
                key=f"department_val_{nonce}",
            )

        r2_c1, r2_c2, r2_c3 = st.columns(3, gap="medium")
        with r2_c1:
            level_val = st.selectbox(
                "Job Level (Junior=0, Lead=1, Mid=2, Senior=3, Manager=4)",
                options=UI_CATEGORY_OPTIONS["Job_Level"],
                key=f"job_level_val_{nonce}",
            )
        with r2_c2:
            remote_val = st.selectbox(
                "Remote Work (No=0, Yes=1)",
                options=UI_CATEGORY_OPTIONS["Remote_Work"],
                key=f"remote_val_{nonce}",
            )
        with r2_c3:
            city_val = st.selectbox(
                "City (Chennai=0, Delhi=1, Hyderabad=2, Mumbai=3)",
                options=UI_CATEGORY_OPTIONS["City"],
                key=f"city_val_{nonce}",
            )

        r3_c1, r3_c2, r3_c3, r3_c4, r3_c5 = st.columns(5, gap="small")
        with r3_c1:
            perf_val = st.selectbox(
                "Performance Rating",
                options=[1, 2, 3, 4, 5],
                key=f"perf_val_{nonce}",
            )
        with r3_c2:
            cert_val = st.number_input(
                "Certifications",
                min_value=0,
                max_value=15,
                step=1,
                key=f"cert_val_{nonce}",
            )
        with r3_c3:
            overtime_val = st.number_input(
                "Overtime Hours",
                min_value=0,
                max_value=100,
                step=1,
                key=f"overtime_val_{nonce}",
            )
        with r3_c4:
            tenure_val = st.number_input(
                "Company Tenure (Yrs)",
                min_value=0,
                max_value=35,
                step=1,
                key=f"tenure_val_{nonce}",
            )
        with r3_c5:
            projects_val = st.number_input(
                "Projects Completed",
                min_value=0,
                max_value=50,
                step=1,
                key=f"projects_val_{nonce}",
            )

        st.markdown('<hr class="studio-divider" />', unsafe_allow_html=True)

        # PRIMARY CTA — GENERATE SALARY PREDICTION
        _, cta_col, _ = st.columns([1, 2, 1])
        with cta_col:
            predict_clicked = st.button(
                "GENERATE SALARY PREDICTION",
                type="primary",
                width="stretch",
            )

        if predict_clicked or st.session_state.get("prediction_result") is not None:
            employee_payload = {
                "Age": int(age_val),
                "Gender": gender_val,
                "Education": edu_val,
                "Experience_Years": float(exp_val),
                "Department": dept_val,
                "Job_Level": level_val,
                "Performance_Rating": int(perf_val),
                "Certifications": int(cert_val),
                "Overtime_Hours": int(overtime_val),
                "Remote_Work": remote_val,
                "City": city_val,
                "Company_Tenure": int(tenure_val),
                "Projects_Completed": int(projects_val),
                "Skill_Score": float(skill_val),
            }
            try:
                predicted_lpa = predict_salary(employee_payload, artifact=artifact)
                avg_lpa = float(meta["average_salary_lpa"])
                diff_lpa = round(predicted_lpa - avg_lpa, 2)
                st.session_state["prediction_result"] = {
                    "details": employee_payload,
                    "predicted_lpa": predicted_lpa,
                    "average_lpa": avg_lpa,
                    "diff_lpa": diff_lpa,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    # SECTION C — PREMIUM SALARY RESULT VISUALIZATION
    result = st.session_state.get("prediction_result")
    if result is not None:
        st.markdown("<div style='height: 1.4rem;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            res_left, res_right = st.columns([3, 2])
            with res_left:
                st.markdown(
                    f"""
                    <h2 class="studio-section-title">YOUR SALARY PREDICTION</h2>
                    <p class="studio-section-desc">
                      Estimated by {selected_metrics['model_name']} · Dataset: {meta['source_name']}
                    </p>
                    """,
                    unsafe_allow_html=True,
                )
            with res_right:
                act1, act2 = st.columns(2, gap="small")
                with act1:
                    if st.button("RESET PREDICTION", type="secondary", width="stretch"):
                        init_input_state(bounds, force=True)
                        st.rerun()
                with act2:
                    report_csv = build_report_csv(
                        details=result["details"],
                        predicted_lpa=result["predicted_lpa"],
                        avg_lpa=result["average_lpa"],
                        diff_lpa=result["diff_lpa"],
                        selected_model_name=selected_metrics["model_name"],
                        timestamp_iso=result["timestamp"],
                    )
                    st.download_button(
                        label="DOWNLOAD REPORT",
                        data=report_csv,
                        file_name="salary_prediction_report.csv",
                        mime="text/csv",
                        width="stretch",
                    )

            st.markdown('<hr class="studio-divider" />', unsafe_allow_html=True)

            gauge_col, details_col = st.columns([5, 7], gap="large")
            with gauge_col:
                salary_fig = build_salary_gauge_figure(
                    predicted_lpa=result["predicted_lpa"],
                    min_lpa=meta["min_salary_lpa"],
                    max_lpa=meta["max_salary_lpa"],
                )
                st.plotly_chart(
                    salary_fig,
                    width="stretch",
                    config={"displayModeBar": False, "staticPlot": True},
                )

            with details_col:
                diff = result["diff_lpa"]
                if diff > 0.005:
                    diff_class = "up"
                    diff_text = f"▲ +₹{abs(diff):.2f} LPA"
                    diff_note = "Above dataset average"
                elif diff < -0.005:
                    diff_class = "down"
                    diff_text = f"▼ -₹{abs(diff):.2f} LPA"
                    diff_note = "Below dataset average"
                else:
                    diff_class = ""
                    diff_text = "₹0.00 LPA"
                    diff_note = "Matches dataset average"

                d = result["details"]
                exp_disp = (
                    f"{int(d['Experience_Years'])}"
                    if float(d["Experience_Years"]).is_integer()
                    else f"{d['Experience_Years']:.1f}"
                )
                skill_disp = (
                    f"{int(d['Skill_Score'])}"
                    if float(d["Skill_Score"]).is_integer()
                    else f"{d['Skill_Score']:.1f}"
                )

                st.markdown(
                    f"""
                    <div class="metric-grid">
                      <div class="metric-card">
                        <div class="metric-label">Predicted Salary</div>
                        <div class="metric-value">₹{result['predicted_lpa']:.2f} LPA</div>
                        <div class="metric-sub">Model inference output</div>
                      </div>
                      <div class="metric-card">
                        <div class="metric-label">Dataset Average Salary</div>
                        <div class="metric-value">₹{result['average_lpa']:.2f} LPA</div>
                        <div class="metric-sub">Actual target mean</div>
                      </div>
                      <div class="metric-card">
                        <div class="metric-label">Difference From Average</div>
                        <div class="metric-value {diff_class}">{diff_text}</div>
                        <div class="metric-sub">{diff_note}</div>
                      </div>
                    </div>
                    <div style="font-family:'Outfit',sans-serif;font-size:0.9rem;font-weight:700;color:#FFFFFF;margin-bottom:0.4rem;">
                      Submitted Employee Profile (14 Notebook Features)
                    </div>
                    <div class="profile-grid">
                      <div class="profile-item"><span class="profile-k">Age</span><span class="profile-v">{d['Age']} yrs</span></div>
                      <div class="profile-item"><span class="profile-k">Gender</span><span class="profile-v">{d['Gender']} ({NOTEBOOK_CATEGORY_MAPS['Gender'][d['Gender']]})</span></div>
                      <div class="profile-item"><span class="profile-k">Education</span><span class="profile-v">{d['Education']} ({NOTEBOOK_CATEGORY_MAPS['Education'][d['Education']]})</span></div>
                      <div class="profile-item"><span class="profile-k">Experience</span><span class="profile-v">{exp_disp} yrs</span></div>
                      <div class="profile-item"><span class="profile-k">Department</span><span class="profile-v">{d['Department']} ({NOTEBOOK_CATEGORY_MAPS['Department'][d['Department']]})</span></div>
                      <div class="profile-item"><span class="profile-k">Job Level</span><span class="profile-v">{d['Job_Level']} ({NOTEBOOK_CATEGORY_MAPS['Job_Level'][d['Job_Level']]})</span></div>
                      <div class="profile-item"><span class="profile-k">Performance Rating</span><span class="profile-v">{d.get('Performance_Rating', 1)}</span></div>
                      <div class="profile-item"><span class="profile-k">Certifications</span><span class="profile-v">{d.get('Certifications', 4)}</span></div>
                      <div class="profile-item"><span class="profile-k">Overtime Hours</span><span class="profile-v">{d.get('Overtime_Hours', 15)} hrs</span></div>
                      <div class="profile-item"><span class="profile-k">Remote Work</span><span class="profile-v">{d['Remote_Work']} ({NOTEBOOK_CATEGORY_MAPS['Remote_Work'][d['Remote_Work']]})</span></div>
                      <div class="profile-item"><span class="profile-k">City</span><span class="profile-v">{d['City']} ({NOTEBOOK_CATEGORY_MAPS['City'][d['City']]})</span></div>
                      <div class="profile-item"><span class="profile-k">Company Tenure</span><span class="profile-v">{d.get('Company_Tenure', 6)} yrs</span></div>
                      <div class="profile-item"><span class="profile-k">Projects Completed</span><span class="profile-v">{d.get('Projects_Completed', 5)}</span></div>
                      <div class="profile-item"><span class="profile-k">Skill Score</span><span class="profile-v">{skill_disp}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown('<hr class="studio-divider" />', unsafe_allow_html=True)

            xgb_note = (
                ""
                if artifact.get("xgboost_available", True)
                else " · Note: xgboost package not installed; evaluated the remaining 3 models."
            )
            rows_html = []
            for ev in artifact["all_evaluations"]:
                row_cls = "selected-row" if ev["is_selected"] else ""
                sel_html = (
                    "<span style='color:#19C991;font-weight:600;'>Active (employee_selary.pkl)</span>"
                    if ev["is_selected"]
                    else "<span style='color:#A8B0C5;'>Notebook Benchmark</span>"
                )
                rows_html.append(
                    f"<tr class='{row_cls}'>"
                    f"<td>{ev['model_name']}</td>"
                    f"<td class='num'>{ev['r2_score']:.4f}</td>"
                    f"<td class='num'>₹{ev['mae']:.3f}</td>"
                    f"<td class='num'>₹{ev['rmse']:.3f}</td>"
                    f"<td style='text-align:right;'>{sel_html}</td>"
                    f"</tr>"
                )

            st.markdown(
                f"""
                <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:0.5rem;">
                  <div style="font-family:'Outfit',sans-serif;font-size:1rem;font-weight:700;color:#FFFFFF;">
                    Model Evaluation Information (employee_selary.ipynb Cells 63–78)
                  </div>
                  <div style="font-size:0.78rem;color:#A8B0C5;">
                    80/20 train_test_split (random_state=42, {meta['test_count']} test rows) · Deployed model: xg_final (employee_selary.pkl){xgb_note}
                  </div>
                </div>
                <table class="eval-table">
                  <thead>
                    <tr>
                      <th>Regression Model</th>
                      <th class="num">R² Score</th>
                      <th class="num">MAE (LPA)</th>
                      <th class="num">RMSE (LPA)</th>
                      <th style="text-align:right;">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {''.join(rows_html)}
                  </tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )


def _is_port_in_use(port: int = 8501) -> bool:
    for family, addr in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.25)
                if sock.connect_ex((addr, port)) == 0:
                    return True
        except Exception:
            continue
    return False


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        _ctx = get_script_run_ctx()
    except Exception:
        _ctx = None

    if _ctx is not None:
        main()
    else:
        # Running in interactive notebook/console (where __file__ is not defined) or `python app.py`
        _active_port = 8501 if _is_port_in_use(8501) else (8502 if _is_port_in_use(8502) else None)
        if _active_port is None and APP_FILE.exists():
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(APP_FILE),
                    "--server.port",
                    "8501",
                    "--server.headless",
                    "true",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _active_port = 8501

        _sample_pred = predict_salary(NOTEBOOK_CELL82_DEFAULTS)
        print(f"Backend Connected: {BACKEND_PKL_PATH.name} | Sample Prediction (Cell 82): ₹{_sample_pred:.2f} LPA")
        if _active_port is not None:
            print(f"Streamlit UI is live at: http://localhost:{_active_port}")
