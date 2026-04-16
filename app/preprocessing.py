from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "diagnostic"
GROUP_COL = "patient_id"
IMAGE_COL = "img_id"

DROP_COLS = ["diagnostic", "img_id", "patient_id", "lesion_id"]

EXPECTED_METADATA_FIELDS = [
    "smoke",
    "drink",
    "background_father",
    "background_mother",
    "age",
    "pesticide",
    "gender",
    "skin_cancer_history",
    "cancer_history",
    "has_piped_water",
    "has_sewage_system",
    "fitspatrick",
    "region",
    "diameter_1",
    "diameter_2",
    "itch",
    "grew",
    "hurt",
    "changed",
    "bleed",
    "elevation",
    "biopsed",
]

ARTIFACTS_FILENAME = "preprocessing_bundle.joblib"
META_FILENAME = "preprocessing_meta.json"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class PreprocessingArtifacts:
    preprocessor: ColumnTransformer
    feature_cols: list[str]
    cat_cols: list[str]
    num_cols: list[str]
    diameter_cols: list[str]
    other_num_cols: list[str]
    outlier_scaler: Pipeline | None
    outlier_detector: IsolationForest | None
    classes: list[str]


def stratified_group_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match notebook split: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(df, y=df[TARGET_COL], groups=df[GROUP_COL]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def _build_preprocessor(feature_df: pd.DataFrame) -> tuple[ColumnTransformer, dict[str, list[str]]]:
    cat_cols = feature_df.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

    diameter_cols = [c for c in ["diameter_1", "diameter_2"] if c in num_cols]
    other_num_cols = [c for c in num_cols if c not in diameter_cols]

    diameter_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    other_numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if diameter_cols:
        transformers.append(("num_diam", diameter_pipeline, diameter_cols))
    if other_num_cols:
        transformers.append(("num_other", other_numeric_pipeline, other_num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    groups = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "diameter_cols": diameter_cols,
        "other_num_cols": other_num_cols,
    }
    return preprocessor, groups


def fit_artifacts(metadata_df: pd.DataFrame) -> tuple[PreprocessingArtifacts, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Fit preprocessing artifacts exactly with notebook conventions."""
    required = {TARGET_COL, GROUP_COL}
    missing_required = [c for c in required if c not in metadata_df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns for fit: {missing_required}")

    train_df, test_df = stratified_group_split(metadata_df)
    feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

    preprocessor, groups = _build_preprocessor(train_df[feature_cols])
    cat_cols = groups["cat_cols"]

    x_train_raw = train_df[feature_cols].copy()
    x_test_raw = test_df[feature_cols].copy()

    if cat_cols:
        x_train_raw[cat_cols] = x_train_raw[cat_cols].astype(str)
        x_test_raw[cat_cols] = x_test_raw[cat_cols].astype(str)

    x_train = preprocessor.fit_transform(x_train_raw)
    x_test = preprocessor.transform(x_test_raw)

    outlier_scaler = None
    outlier_detector = None
    if groups["diameter_cols"]:
        outlier_scaler = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        x_train_num = outlier_scaler.fit_transform(x_train_raw[groups["diameter_cols"]])
        outlier_detector = IsolationForest(contamination=0.05, random_state=42)
        outlier_detector.fit(x_train_num)

    classes = sorted(train_df[TARGET_COL].dropna().astype(str).unique().tolist())

    artifacts = PreprocessingArtifacts(
        preprocessor=preprocessor,
        feature_cols=feature_cols,
        cat_cols=groups["cat_cols"],
        num_cols=groups["num_cols"],
        diameter_cols=groups["diameter_cols"],
        other_num_cols=groups["other_num_cols"],
        outlier_scaler=outlier_scaler,
        outlier_detector=outlier_detector,
        classes=classes,
    )
    return artifacts, train_df, test_df, x_train, x_test


def save_artifacts(artifacts: PreprocessingArtifacts, out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    bundle_path = out_path / ARTIFACTS_FILENAME
    joblib.dump(artifacts, bundle_path)

    meta = {
        "saved_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "target_col": TARGET_COL,
        "group_col": GROUP_COL,
        "drop_cols": DROP_COLS,
        "feature_cols": artifacts.feature_cols,
        "cat_cols": artifacts.cat_cols,
        "num_cols": artifacts.num_cols,
        "diameter_cols": artifacts.diameter_cols,
        "other_num_cols": artifacts.other_num_cols,
        "classes": artifacts.classes,
    }
    with (out_path / META_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return bundle_path


def load_artifacts(artifacts_dir: str | Path) -> PreprocessingArtifacts:
    path = Path(artifacts_dir) / ARTIFACTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessing artifact not found at '{path}'. Run scripts/fit_preprocessing.py first."
        )
    artifacts = joblib.load(path)
    if isinstance(artifacts, PreprocessingArtifacts):
        return artifacts

    if isinstance(artifacts, dict):
        required = {
            "preprocessor",
            "feature_cols",
            "cat_cols",
            "num_cols",
            "diameter_cols",
            "other_num_cols",
            "outlier_scaler",
            "outlier_detector",
            "classes",
        }
        missing = [k for k in required if k not in artifacts]
        if missing:
            raise TypeError(
                "Joblib artifact dict is missing required keys: "
                f"{missing}. Expected keys: {sorted(required)}"
            )

        return PreprocessingArtifacts(
            preprocessor=artifacts["preprocessor"],
            feature_cols=list(artifacts["feature_cols"]),
            cat_cols=list(artifacts["cat_cols"]),
            num_cols=list(artifacts["num_cols"]),
            diameter_cols=list(artifacts["diameter_cols"]),
            other_num_cols=list(artifacts["other_num_cols"]),
            outlier_scaler=artifacts["outlier_scaler"],
            outlier_detector=artifacts["outlier_detector"],
            classes=list(artifacts["classes"]),
        )

    raise TypeError(f"Unexpected artifact type: {type(artifacts)}")


def _coerce_value(value: Any) -> Any:
    if value is None:
        return np.nan
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed == "":
            return np.nan
        if trimmed.lower() in {"nan", "none", "null"}:
            return np.nan
        return trimmed
    return value


def metadata_to_dataframe(metadata: dict[str, Any], feature_cols: list[str]) -> pd.DataFrame:
    row = {col: np.nan for col in feature_cols}
    for key, value in metadata.items():
        if key in row:
            row[key] = _coerce_value(value)
    return pd.DataFrame([row], columns=feature_cols)


def preprocess_metadata_row(
    metadata: dict[str, Any],
    artifacts: PreprocessingArtifacts,
) -> np.ndarray:
    row_df = metadata_to_dataframe(metadata, artifacts.feature_cols)
    if artifacts.num_cols:
        for col in artifacts.num_cols:
            if col in row_df.columns:
                row_df[col] = pd.to_numeric(row_df[col], errors="coerce")
    if artifacts.cat_cols:
        row_df[artifacts.cat_cols] = row_df[artifacts.cat_cols].astype(str)
    transformed = artifacts.preprocessor.transform(row_df)
    return np.asarray(transformed, dtype=np.float32)


def outlier_flag_for_metadata(
    metadata: dict[str, Any],
    artifacts: PreprocessingArtifacts,
) -> int | None:
    if not artifacts.diameter_cols or artifacts.outlier_scaler is None or artifacts.outlier_detector is None:
        return None

    row_df = metadata_to_dataframe(metadata, artifacts.diameter_cols)
    for col in artifacts.diameter_cols:
        if col in row_df.columns:
            row_df[col] = pd.to_numeric(row_df[col], errors="coerce")
    scaled = artifacts.outlier_scaler.transform(row_df)
    prediction = int(artifacts.outlier_detector.predict(scaled)[0])
    return prediction


def preprocess_image_eval(image: Image.Image) -> np.ndarray:
    """Inference-time image preprocessing matching notebook common_eval_tfms."""
    img = image.convert("RGB")
    resampling = getattr(Image, "Resampling", Image)
    img = img.resize((224, 224), resampling.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)
    arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)
