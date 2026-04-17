from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import onnxruntime as ort
import requests


DEFAULT_MODEL_DIR = Path("artifacts") / "models"
DEFAULT_MODEL_FILENAME = "model.onnx"
DEFAULT_GOOGLE_DRIVE_URL = "https://drive.google.com/file/d/1l594HdeuFiWKug3Dqew-GBUzq6FH9inH/view?usp=sharing"


@dataclass
class OnnxClassifier:
    session: ort.InferenceSession
    model_path: Path
    class_names: list[str]


@dataclass
class ClassificationResult:
    class_index: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]


def _is_truthy(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _extract_google_drive_file_id(url: str) -> str | None:
    parsed = urlparse(url)
    query_id = parse_qs(parsed.query).get("id")
    if query_id and query_id[0].strip():
        return query_id[0].strip()

    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    return None


def _stream_download(response: requests.Response, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file_obj.write(chunk)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {output_path}")


def _download_from_google_drive(file_id: str, output_path: Path, timeout: int = 120) -> None:
    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"

    response = session.get(base_url, params={"id": file_id}, stream=True, timeout=timeout)
    response.raise_for_status()

    warning_token = next((value for key, value in response.cookies.items() if key.startswith("download_warning")), None)
    if warning_token:
        response.close()
        response = session.get(
            base_url,
            params={"id": file_id, "confirm": warning_token},
            stream=True,
            timeout=timeout,
        )
        response.raise_for_status()

    _stream_download(response, output_path)
    response.close()


def _download_from_url(url: str, output_path: Path, timeout: int = 120) -> None:
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    _stream_download(response, output_path)
    response.close()


def _resolve_model_path() -> Path:
    configured_path = os.getenv("ONNX_MODEL_PATH")
    if configured_path:
        return Path(configured_path)

    model_dir = Path(os.getenv("ONNX_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    model_filename = os.getenv("ONNX_MODEL_FILENAME", DEFAULT_MODEL_FILENAME)
    return model_dir / model_filename


def _resolve_class_names(classes_from_artifacts: list[str] | None, num_classes: int) -> list[str]:
    classes: list[str] = []
    raw = os.getenv("ONNX_CLASS_NAMES", "")
    if raw.strip():
        classes = [part.strip() for part in raw.split(",") if part.strip()]
    elif classes_from_artifacts:
        classes = list(classes_from_artifacts)

    if len(classes) < num_classes:
        classes.extend([f"class_{idx}" for idx in range(len(classes), num_classes)])

    return classes[:num_classes]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_vals = np.exp(shifted)
    total = np.sum(exp_vals)
    if total <= 0:
        return np.full_like(values, 1.0 / values.size)
    return exp_vals / total


def _shape_rank(shape: list[int | str | None]) -> int | None:
    if not shape:
        return None
    return len(shape)


def _is_nhwc_shape(shape: list[int | str | None]) -> bool:
    if len(shape) != 4:
        return False
    channel_dim = shape[3]
    return channel_dim in (3, "3", None, "None")


def _coerce_to_input_shape(array: np.ndarray, expected_shape: list[int | str | None]) -> np.ndarray:
    coerced = np.asarray(array, dtype=np.float32)
    rank = _shape_rank(expected_shape)

    if rank == 4 and coerced.ndim == 3:
        coerced = np.expand_dims(coerced, axis=0)
    elif rank == 3 and coerced.ndim == 4 and coerced.shape[0] == 1:
        coerced = coerced[0]
    elif rank == 2 and coerced.ndim == 1:
        coerced = coerced.reshape(1, -1)
    elif rank == 1 and coerced.ndim == 2 and coerced.shape[0] == 1:
        coerced = coerced.reshape(-1)

    if rank == 4 and coerced.ndim == 4 and _is_nhwc_shape(expected_shape) and coerced.shape[1] == 3:
        coerced = np.transpose(coerced, (0, 2, 3, 1))

    return coerced.astype(np.float32)


def _build_feed_dict(
    session: ort.InferenceSession,
    image_tensor: np.ndarray,
    metadata_vector: np.ndarray,
) -> dict[str, np.ndarray]:
    image_batch = np.asarray(image_tensor, dtype=np.float32)
    if image_batch.ndim == 3:
        image_batch = np.expand_dims(image_batch, axis=0)

    metadata_batch = np.asarray(metadata_vector, dtype=np.float32)
    if metadata_batch.ndim == 1:
        metadata_batch = metadata_batch.reshape(1, -1)

    session_inputs = session.get_inputs()
    if not session_inputs:
        raise RuntimeError("ONNX model has no declared inputs.")

    if len(session_inputs) == 1:
        single_input = session_inputs[0]
        rank = _shape_rank(single_input.shape)
        source = image_batch if (rank is not None and rank >= 3) else metadata_batch
        return {single_input.name: _coerce_to_input_shape(source, single_input.shape)}

    feed_dict: dict[str, np.ndarray] = {}
    for input_info in session_inputs:
        normalized_name = re.sub(r"[^a-z0-9]", "", input_info.name.lower())
        rank = _shape_rank(input_info.shape)

        if any(token in normalized_name for token in ("meta", "tab", "clinical", "feature")):
            feed_dict[input_info.name] = _coerce_to_input_shape(metadata_batch, input_info.shape)
            continue

        if any(token in normalized_name for token in ("img", "image", "pixel", "vision")):
            feed_dict[input_info.name] = _coerce_to_input_shape(image_batch, input_info.shape)
            continue

        if rank is not None and rank >= 3:
            feed_dict[input_info.name] = _coerce_to_input_shape(image_batch, input_info.shape)
        else:
            feed_dict[input_info.name] = _coerce_to_input_shape(metadata_batch, input_info.shape)

    return feed_dict


def _scores_from_output(raw_output: np.ndarray) -> np.ndarray:
    output = np.asarray(raw_output)
    if output.ndim == 0:
        return np.array([float(output)], dtype=np.float32)
    if output.ndim == 1:
        return output.astype(np.float32)
    flattened = output.reshape(output.shape[0], -1)
    return flattened[0].astype(np.float32)


def ensure_onnx_model() -> Path:
    model_path = _resolve_model_path()
    if model_path.exists():
        return model_path

    if not _is_truthy(os.getenv("ONNX_MODEL_AUTO_DOWNLOAD"), default=True):
        raise FileNotFoundError(
            f"ONNX model not found at '{model_path}' and ONNX_MODEL_AUTO_DOWNLOAD is disabled."
        )

    source_url = os.getenv("ONNX_MODEL_DRIVE_URL", DEFAULT_GOOGLE_DRIVE_URL)
    file_id = _extract_google_drive_file_id(source_url)

    if file_id:
        _download_from_google_drive(file_id, model_path)
    else:
        _download_from_url(source_url, model_path)

    return model_path


def load_onnx_classifier(classes_from_artifacts: list[str] | None = None) -> OnnxClassifier:
    model_path = ensure_onnx_model()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    output_info = session.get_outputs()
    if not output_info:
        raise RuntimeError("ONNX model has no outputs.")

    output_shape = output_info[0].shape
    output_rank = _shape_rank(output_shape)
    if output_rank == 0:
        num_classes = 1
    else:
        last_dim = output_shape[-1]
        num_classes = int(last_dim) if isinstance(last_dim, int) and last_dim > 0 else 0

    if num_classes <= 0:
        num_classes = max(1, len(classes_from_artifacts or []))

    class_names = _resolve_class_names(classes_from_artifacts, num_classes)
    return OnnxClassifier(session=session, model_path=model_path, class_names=class_names)


def classify(
    classifier: OnnxClassifier,
    image_tensor: np.ndarray,
    metadata_vector: np.ndarray,
) -> ClassificationResult:
    feed_dict = _build_feed_dict(classifier.session, image_tensor=image_tensor, metadata_vector=metadata_vector)
    outputs = classifier.session.run(None, feed_dict)
    if not outputs:
        raise RuntimeError("ONNX inference returned no outputs.")

    scores = _scores_from_output(outputs[0])

    if scores.size == 1 and len(classifier.class_names) == 2:
        positive = 1.0 / (1.0 + np.exp(-scores[0]))
        probabilities = np.array([1.0 - positive, positive], dtype=np.float32)
    elif scores.size == 1:
        probabilities = np.array([1.0], dtype=np.float32)
    else:
        if np.all(scores >= 0.0) and np.all(scores <= 1.0) and np.isclose(float(np.sum(scores)), 1.0, atol=1e-3):
            probabilities = scores
        else:
            probabilities = _softmax(scores)

    class_names = classifier.class_names
    if len(class_names) < probabilities.size:
        class_names = class_names + [f"class_{idx}" for idx in range(len(class_names), probabilities.size)]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])
    probability_map = {class_names[idx]: float(probabilities[idx]) for idx in range(probabilities.size)}

    return ClassificationResult(
        class_index=predicted_index,
        class_name=predicted_class,
        confidence=confidence,
        probabilities=probability_map,
    )
