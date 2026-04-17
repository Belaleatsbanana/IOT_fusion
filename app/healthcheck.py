from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.inference import load_onnx_classifier
from app.preprocessing import load_artifacts


DEFAULT_ARTIFACTS_DIR = Path("artifacts") / "preprocessing"


def _load_local_env() -> None:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def run_healthcheck() -> dict[str, Any]:
    artifacts_dir = os.getenv("PREPROCESS_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR))
    artifacts = load_artifacts(artifacts_dir)
    classifier = load_onnx_classifier(classes_from_artifacts=artifacts.classes)

    return {
        "ok": True,
        "artifacts_dir": artifacts_dir,
        "feature_count": len(artifacts.feature_cols),
        "classes": artifacts.classes,
        "onnx_model_path": str(classifier.model_path),
        "onnx_inputs": [input_info.name for input_info in classifier.session.get_inputs()],
        "onnx_outputs": [output_info.name for output_info in classifier.session.get_outputs()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Application readiness healthcheck")
    parser.add_argument("--quiet", action="store_true", help="Print output only on failure")
    args = parser.parse_args()

    _load_local_env()

    try:
        result = run_healthcheck()
        if not args.quiet:
            print(json.dumps(result, indent=2))
    except Exception as exc:  # noqa: BLE001
        error = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(error, indent=2))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
