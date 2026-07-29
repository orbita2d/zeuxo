# zeuxo

Tooling and training for the zeuxo chess engine — training scripts, data
pipelines, and hyperparameter tuning.

## training/

- `training.py` — self-contained (PEP 723 inline deps) JAX/Flax NNUE-style
  trainer. Run locally with `uv run training.py`; hyperparameters and paths come
  from environment variables (see the top of the file).
- `zeuxo-training-skypilot.yaml` — launches training on RunPod, joins the
  tailnet, and logs to MLflow. `sky launch zeuxo-training-skypilot.yaml --secret TS_AUTHKEY --env TRAIN_BOARDS=...`.

Required env vars: `TRAIN_FILE`, `CHECKPOINT_PATH`, `TRAIN_BOARDS`.
