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

## data/

Scala 3 / fs2 pipeline (forked from admete-meta) that turns Lichess `.pgn.zst`
dumps into the training parquet. Unlike the admete pipeline it does no
quiescence — zeuxo is aimed at search-free eval, so positions are stored
as-played with the raw Stockfish eval (mates mapped to ±10000cp). Boards are
encoded in Scala (`Encoding.scala`) from the side-to-move's perspective with
dedicated tokens for the en-passant-capturable pawn and rooks with castling
rights; the engine-side encoder must match this exactly.

```
sbt "run <positionLimit> <pgnDir> <outputDir> [minElo] [debugFen]"
```

Games split 95/5 into train/test partitions (whole games, not positions).
Pass `true` as the fifth argument to also store each position's FEN for
spot-checking.

For long builds, `Dockerfile` packages the pipeline; configure with `PGN_DIR`
(default `/chess/games`), `OUTPUT_DIR` (default `/chess/training/data`, must be
empty), `POSITIONS`, `MIN_ELO`, `DEBUG_FEN`, and `JAVA_OPTS`.

