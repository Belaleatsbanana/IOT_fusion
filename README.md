# IOT Fusion Preprocessing UI (Streamlit)

This app extracts preprocessing logic from `exploration.ipynb`, saves it as a Joblib artifact, and serves a Streamlit frontend that:

- accepts an image and metadata,
- applies the same preprocessing,
- runs ONNX inference,
- returns the predicted class and probabilities.

## What is persisted

The script stores a reusable preprocessing bundle in `artifacts/preprocessing/`:

- `preprocessing_bundle.joblib` (fitted `ColumnTransformer` + outlier components + feature schema)
- `preprocessing_meta.json` (human-readable metadata)

The UI only **loads and transforms** with this artifact.

## Preprocessing parity with notebook

Tabular pipeline follows `exploration.ipynb`:

- split with `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
- drop `diagnostic`, `img_id`, `patient_id`, `lesion_id` from feature set
- `diameter_1` and `diameter_2`: median impute + standard scale
- other numeric: median impute only (`age`, `fitspatrick`)
- categorical/bool: constant impute (`Unknown`) + one-hot (`handle_unknown='ignore'`)
- isolation forest for outlier flaging on diameter columns only (`contamination=0.05`, `random_state=42`)

Image inference preprocessing follows notebook eval transform:

- RGB convert
- resize to `224x224`
- normalize using ImageNet mean/std
- CHW float32 output

## ONNX model download

The application can auto-download the ONNX model from Google Drive at startup.

Default link used:

- `https://drive.google.com/file/d/1l594HdeuFiWKug3Dqew-GBUzq6FH9inH/view?usp=sharing`

Configurable environment variables:

- `ONNX_MODEL_AUTO_DOWNLOAD=true|false` (default: `true`)
- `ONNX_MODEL_DRIVE_URL=<google drive or direct URL>`
- `ONNX_MODEL_DIR=<target folder>` (default: `artifacts/models`)
- `ONNX_MODEL_FILENAME=<file name>` (default: `model.onnx`)
- `ONNX_MODEL_PATH=<full path>` (overrides dir+filename)
- `ONNX_CLASS_NAMES=class_a,class_b,...` (optional explicit class names)

## Local run

Install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Save/export preprocessing from `exploration.ipynb` so that
`preprocessing_bundle.joblib` exists.

Then set `.env` (or shell env):

```bash
PREPROCESS_ARTIFACTS_DIR=/absolute/path/to/joblibs/folder
ONNX_MODEL_AUTO_DOWNLOAD=true
ONNX_MODEL_DRIVE_URL=https://drive.google.com/file/d/1l594HdeuFiWKug3Dqew-GBUzq6FH9inH/view?usp=sharing
ONNX_MODEL_DIR=/absolute/path/to/model/dir
ONNX_MODEL_FILENAME=model.onnx
```

Notes:

- `PREPROCESS_ARTIFACTS_DIR` must be a local/mounted filesystem path that contains
  `preprocessing_bundle.joblib`.
- ONNX model is downloaded automatically if not present in the configured location.

Run Streamlit UI:

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

## Streamlit behavior

- Loads `artifacts/preprocessing/preprocessing_bundle.joblib`
- Builds metadata form from artifact feature columns
- Preprocesses image with notebook-compatible eval transform
- Preprocesses metadata via loaded `ColumnTransformer.transform(...)`
- Runs ONNX inference and returns predicted class with confidence/probabilities
- Also shows image tensor and metadata vector previews in UI

## Docker

Build and run:

```bash
docker compose up --build
```

The compose file mounts `./artifacts` into the container. Make sure
`preprocessing_bundle.joblib` exists inside the mounted path.

At container startup, the app also downloads `model.onnx` into
`/app/artifacts/models` unless already present.

## Healthcheck

The container includes a readiness healthcheck that verifies:

- preprocessing artifact can be loaded,
- ONNX model is present/loadable,
- ONNX runtime session exposes valid inputs/outputs.

Run manually:

```bash
python -m app.healthcheck
```

Quiet mode (exit code only):

```bash
python -m app.healthcheck --quiet
```

## GitHub CI

Workflow: `.github/workflows/docker-image.yml`

- builds Docker image on push/PR
- automatically publishes to GitHub Container Registry (GHCR) on push events
- uses branch/sha tags and `latest` for the default branch

Published image format:

- `ghcr.io/<owner>/<repo>:<branch>`
- `ghcr.io/<owner>/<repo>:sha-<commit>`
- `ghcr.io/<owner>/<repo>:latest` (default branch only)
