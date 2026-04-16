# IOT Fusion Preprocessing UI (Streamlit)

This app extracts preprocessing logic from `exploration.ipynb`, saves it as a Joblib artifact, and serves a Streamlit frontend that:

- accepts an image and metadata,
- applies the same preprocessing,
- returns transformed tensors/vectors (model call currently stubbed).

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
```

Notes:

- `PREPROCESS_ARTIFACTS_DIR` must be a local/mounted filesystem path that contains
  `preprocessing_bundle.joblib`.

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
- Returns image tensor and metadata vector previews in UI

## Docker

Build and run:

```bash
docker compose up --build
```

The compose file mounts `./artifacts` into the container. Make sure
`preprocessing_bundle.joblib` exists inside the mounted path.

## GitHub CI

Workflow: `.github/workflows/docker-image.yml`

- builds Docker image on push/PR
- automatically publishes to GitHub Container Registry (GHCR) on push events
- uses branch/sha tags and `latest` for the default branch

Published image format:

- `ghcr.io/<owner>/<repo>:<branch>`
- `ghcr.io/<owner>/<repo>:sha-<commit>`
- `ghcr.io/<owner>/<repo>:latest` (default branch only)
