from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from app.inference import ensure_onnx_model


def main() -> None:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    model_path = ensure_onnx_model()
    print(f"ONNX model ready at: {model_path}")


if __name__ == "__main__":
    main()
