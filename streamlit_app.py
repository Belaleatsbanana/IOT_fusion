from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

from app.inference import OnnxClassifier, classify, load_onnx_classifier
from app.preprocessing import (
    EXPECTED_METADATA_FIELDS,
    PreprocessingArtifacts,
    load_artifacts,
    outlier_flag_for_metadata,
    preprocess_image_eval,
    preprocess_metadata_row,
)


DEFAULT_ARTIFACTS_DIR = Path("artifacts") / "preprocessing"
ENV_FILE = Path(__file__).resolve().parent / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


@st.cache_resource
def _load_artifacts_cached(artifacts_dir: str) -> PreprocessingArtifacts:
    return load_artifacts(artifacts_dir)


@st.cache_resource
def _load_classifier_cached(classes: tuple[str, ...]) -> OnnxClassifier:
    class_list = list(classes)
    return load_onnx_classifier(classes_from_artifacts=class_list)


def _metadata_fields(artifacts: PreprocessingArtifacts | None) -> list[str]:
    if artifacts is not None:
        return artifacts.feature_cols
    return EXPECTED_METADATA_FIELDS


def _render_metadata_inputs(fields: list[str], artifacts: PreprocessingArtifacts | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    num_cols = set(artifacts.num_cols) if artifacts is not None else set()

    col1, col2 = st.columns(2)
    for idx, field in enumerate(fields):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            if field in num_cols:
                metadata[field] = st.text_input(field, value="", placeholder="numeric")
            else:
                metadata[field] = st.text_input(field, value="")
    return metadata


def main() -> None:
    st.set_page_config(page_title="IOT Fusion Inference", page_icon="🧪", layout="wide")

    st.title("IOT Fusion Classification (Streamlit)")
    st.caption("Upload image + metadata to run preprocessing and ONNX inference in one step.")

    artifacts_dir = os.getenv("PREPROCESS_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR))
    artifacts: PreprocessingArtifacts | None = None
    classifier: OnnxClassifier | None = None
    try:
        artifacts = _load_artifacts_cached(artifacts_dir)
        st.success(f"Loaded preprocessing artifact from `{artifacts_dir}`")
    except FileNotFoundError:
        st.error(
            "Preprocessing artifact not found. Save it from `exploration.ipynb`, then set "
            "`PREPROCESS_ARTIFACTS_DIR` to the local/mounted folder containing `preprocessing_bundle.joblib`."
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load preprocessing artifact: {exc}")

    if artifacts is not None:
        try:
            classifier = _load_classifier_cached(tuple(artifacts.classes))
            st.success(f"Loaded ONNX model from `{classifier.model_path}`")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load ONNX classifier: {exc}")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Input image")
        uploaded = st.file_uploader("Upload lesion image", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded is not None:
            st.image(uploaded, caption="Input preview", use_container_width=True)

    with right:
        st.subheader("Metadata")
        st.caption("Leave unknown fields blank.")
        fields = _metadata_fields(artifacts)
        metadata = _render_metadata_inputs(fields, artifacts)

    if st.button("Run classification", type="primary", use_container_width=True):
        if uploaded is None:
            st.warning("Please upload an image.")
            return
        if artifacts is None:
            st.warning("Artifacts are required for metadata transform.")
            return
        if classifier is None:
            st.warning("ONNX model is required for inference.")
            return

        try:
            image = Image.open(uploaded)
            image_tensor = preprocess_image_eval(image)
        except UnidentifiedImageError:
            st.error("Uploaded file is not a valid image.")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Image preprocessing failed: {exc}")
            return

        try:
            metadata_vec = preprocess_metadata_row(metadata, artifacts)
            outlier_flag = outlier_flag_for_metadata(metadata, artifacts)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Metadata preprocessing failed: {exc}")
            return

        try:
            prediction = classify(
                classifier,
                image_tensor=image_tensor,
                metadata_vector=metadata_vec,
                outlier_flag=outlier_flag,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Model inference failed: {exc}")
            return

        st.subheader("Classification result")
        st.success(
            f"Predicted class: **{prediction.class_name}** "
            f"(index={prediction.class_index}, confidence={prediction.confidence:.4f})"
        )
        st.json(
            {
                "predicted_class": prediction.class_name,
                "class_index": prediction.class_index,
                "confidence": round(prediction.confidence, 6),
                "probabilities": {k: round(v, 6) for k, v in prediction.probabilities.items()},
            }
        )

        st.subheader("Preprocessing details")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Image tensor**")
            st.json(
                {
                    "shape": list(image_tensor.shape),
                    "dtype": str(image_tensor.dtype),
                    "sample": np.round(image_tensor.reshape(-1)[:12], 4).tolist(),
                }
            )

        with c2:
            st.markdown("**Metadata vector**")
            st.json(
                {
                    "shape": list(metadata_vec.shape),
                    "dtype": str(metadata_vec.dtype),
                    "sample": np.round(metadata_vec.reshape(-1)[:16], 4).tolist(),
                    "outlier_flag": outlier_flag,
                }
            )


if __name__ == "__main__":
    main()
