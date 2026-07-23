# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "jax[cuda12]~=0.6.0",
#    "flax~=0.10.6",
#    "optax",
#    "orbax",
#    "pandas",
#    "numpy",
#    "scipy",
#    "pyarrow",
#    "tqdm",
#    "mlflow",
#    "psutil",
#    "pynvml",
#   ]
# ///

import logging
import os
from pathlib import Path
import shutil
from typing import Literal

import jax
from jax import numpy as jnp
import numpy as np
import flax
from flax import nnx
import optax
import orbax.checkpoint as ocp
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import mlflow

logger = logging.getLogger(__name__)

# --- board / feature geometry -------------------------------------------------
N_SQUARES = 64
N_PIECE_TYPES = 5  # pawn, knight, bishop, rook, queen (king handled separately)
N_FEATURES = N_SQUARES * N_PIECE_TYPES
LOGISTIC_SCALING = 400.0  # centipawns -> win prob via sigmoid(eval / scaling)

# --- hyperparameters (env-var driven, same pattern as admete) -----------------
HYPERPARAM_ACCUMULATOR_SIZE = int(os.environ.get("HYPERPARAM_ACCUMULATOR_SIZE", 256))
HYPERPARAM_HIDDEN_SIZE = int(os.environ.get("HYPERPARAM_HIDDEN_SIZE", 32))
HYPERPARAM_LEARNING_RATE_PEAK = float(os.environ.get("HYPERPARAM_LEARNING_RATE_PEAK", 4e-3))
HYPERPARAM_LEARNING_RATE_END = float(os.environ.get("HYPERPARAM_LEARNING_RATE_END", 2e-6))
HYPERPARAM_BATCH_SIZE_LOG = int(os.environ.get("HYPERPARAM_BATCH_SIZE_LOG", 14))  # 2^14 = 16384

ITERATIONS = os.environ.get("ITERATIONS", None)
SEED = int(os.environ.get("SEED", 314159))

PARENT_RUN_ID = os.environ.get("PARENT_RUN_ID", None)
RUN_ID = os.environ.get("RUN_ID", None)
MLFLOW_RUN_DESCRIPTION = os.environ.get("MLFLOW_RUN_DESCRIPTION")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "zeuxo-training")

TRAIN_FILE = os.environ.get("TRAIN_FILE")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH")


class ZeuxoModel(nnx.Module):
    """
    Perspective-net evaluator. Two accumulators (side-to-move + opponent) share a
    single feature-transformer weight, are concatenated, and run through a small
    MLP head to a single scalar (centipawn-ish) evaluation.
    """

    def __init__(self, rngs: nnx.Rngs):
        acc = HYPERPARAM_ACCUMULATOR_SIZE
        hidden = HYPERPARAM_HIDDEN_SIZE

        # shared feature transformer applied to each perspective independently
        self.feature_transformer = nnx.Linear(N_FEATURES, acc, rngs=rngs, use_bias=True)
        self.hidden_layer = nnx.Linear(2 * acc, hidden, rngs=rngs, use_bias=True)
        self.output_layer = nnx.Linear(hidden, 1, rngs=rngs, use_bias=True)

    def __call__(self, us: jax.Array, them: jax.Array) -> jax.Array:
        assert us.shape[1] == N_FEATURES
        assert us.shape == them.shape

        acc_us = self.feature_transformer(us)
        acc_them = self.feature_transformer(them)
        # clipped relu keeps the accumulator quantisation-friendly (NNUE convention)
        x = jnp.concatenate([acc_us, acc_them], axis=-1)
        x = jnp.clip(x, 0.0, 1.0)
        x = nnx.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return jnp.reshape(x, (-1,))


def normalise_eval(eval: jax.Array, logistic_scaling: float) -> jax.Array:
    return nnx.sigmoid(eval / logistic_scaling)


def loss_fn(pred: jax.Array, labels: jax.Array) -> jax.Array:
    # binary cross entropy between predicted and target win probabilities
    eps = 1e-7
    assert pred.shape == labels.shape
    labels = jnp.clip(labels, eps, 1 - eps)
    pred = jnp.clip(pred, eps, 1 - eps)
    return -jnp.mean(labels * jnp.log(pred) + (1 - labels) * jnp.log(1 - pred))


@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch: dict[str, jax.Array]):
    def loss(model: ZeuxoModel):
        pred = model(batch["f_us"], batch["f_them"])
        return loss_fn(nnx.sigmoid(pred), batch["label"])

    grads = nnx.grad(loss)(model)
    loss_val = loss(model)
    optimizer.update(grads)
    metrics.update(loss=loss_val)


@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch: dict[str, jax.Array]):
    pred = nnx.sigmoid(model(batch["f_us"], batch["f_them"]))
    label = batch["label"]
    metrics.update(
        loss=loss_fn(pred, label),
        winning_ratio=jnp.mean(pred > 0.8),
        losing_ratio=jnp.mean(pred < 0.2),
    )


@jax.jit
def unpack_features(features: jax.Array) -> dict[str, jax.Array]:
    """
    Turn a (batch, 64) board of piece codes into side-to-move and opponent
    half-kp-free feature planes.

    Encoding (adjust to match your data generator):
      1..5   -> our pawn..queen,   6  -> our king
      9..13  -> their pawn..queen, 14 -> their king
    The opponent perspective is the vertically-flipped board.
    """
    us_codes = [1, 2, 3, 4, 5]
    them_codes = [9, 10, 11, 12, 13]

    flipped = jnp.flip(features.reshape(-1, 8, 8), axis=1).reshape(-1, 64)

    f_us = jnp.concat([jnp.array(features == c, dtype=jnp.int8) for c in us_codes], axis=1)
    f_them = jnp.concat([jnp.array(flipped == c, dtype=jnp.int8) for c in them_codes], axis=1)
    return {"f_us": f_us, "f_them": f_them}


class FileSource:
    """
    Parquet-backed loader partitioned on setType={train,test,validation}.
    Schema: features (list<int8>[64]), eval (int32), setType (string).
    """

    def __init__(self, path: Path, logistic_scaling: float):
        feature_length = N_SQUARES
        schema = pa.schema([
            pa.field("features", pa.list_(pa.int8(), feature_length)),
            pa.field("eval", pa.int32()),
            pa.field("setType", pa.string()),
        ])
        partition = pds.partitioning(pa.schema([pa.field("setType", pa.string(), nullable=False)]))

        # cap in-memory rows so a big dataset doesn't blow the box (~80 GiB budget)
        self.max_rows = 24 * (1 << 30) // (feature_length + 4)
        self.ds = pds.dataset(path, partitioning=partition, schema=schema)
        self.path = path
        self.logistic_scaling = logistic_scaling

        logger.info(f"Initialised {self.ds.count_rows():,} rows from {path}")
        for split in ("train", "test", "validation"):
            logger.info(f"{split:>10} samples: {self.samples(split):>12,}")

        self._training_data = self.ds.head(self.samples("train"), filter=pds.field("setType") == "setType=train")
        self.ds_jax = {"test": self.load_to_device(
            self.ds.head(self.samples("test"), filter=pds.field("setType") == "setType=test")
        )}
        logger.info(f"Test set on device {self.ds_jax['test']['features'].device}")

    def __len__(self):
        return self.ds.count_rows()

    def samples(self, dataset: Literal["test", "train", "validation"]) -> int:
        assert dataset in ("test", "train", "validation")
        max_rows = min(self.max_rows, self.ds.count_rows())
        n = self.ds.count_rows(filter=pds.field("setType") == f"setType={dataset}")
        return int(n / self.ds.count_rows() * max_rows)

    def load_to_device(self, table: pa.Table) -> dict[str, jax.Array]:
        features = jnp.from_dlpack(table.column("features").combine_chunks().values).reshape(-1, 64)
        arr = {
            "features": features,
            "label": normalise_eval(jnp.from_dlpack(table.column("eval").combine_chunks()), self.logistic_scaling),
        }
        return jax.device_put(arr, device=jax.devices()[0])

    def batched(self, batch_size: int, *, dataset: Literal["test", "train", "validation"], repeat: bool = True):
        if dataset == "test":
            yield from self._batched_from_device(batch_size, dataset="test")
        else:
            yield from self._batched_lazy(batch_size, dataset=dataset, repeat=repeat)

    def _batched_from_device(self, batch_size: int, dataset: str):
        ds = self.ds_jax[dataset]
        chunk_len = len(ds["features"])
        for i in range(0, chunk_len - batch_size + 1, batch_size):
            yield {
                **unpack_features(ds["features"][i:i + batch_size]),
                "label": ds["label"][i:i + batch_size],
            }

    def _batched_lazy(self, batch_size: int, *, dataset: str, repeat: bool):
        if dataset == "train":
            chunk = self._training_data
        elif dataset == "validation":
            chunk = self.ds.head(self.samples("validation"), filter=pds.field("setType") == "setType=validation")
        else:
            raise ValueError(dataset)

        while True:
            for i in range(0, chunk.num_rows - batch_size + 1, batch_size):
                arr = self.load_to_device(chunk.slice(offset=i, length=batch_size))
                yield {**unpack_features(arr["features"]), "label": arr["label"]}
            if not repeat:
                break

    def clear_in_memory(self):
        self._training_data = None


def main() -> None:
    assert TRAIN_FILE is not None, "TRAIN_FILE environment variable is not set."
    assert CHECKPOINT_PATH is not None, "CHECKPOINT_PATH environment variable is not set."
    assert ITERATIONS is not None, "ITERATIONS environment variable is not set."

    train_file = Path(TRAIN_FILE)
    assert train_file.exists(), f"Train file {train_file} does not exist."
    iterations = int(ITERATIONS)
    assert iterations > 0, f"Iterations must be positive, not {iterations}."

    checkpoint_root = Path(CHECKPOINT_PATH)
    assert checkpoint_root.exists(), f"Checkpoint path {checkpoint_root} does not exist."
    checkpoint_path = (checkpoint_root / pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")).resolve()
    checkpoint_path.mkdir()

    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(checkpoint_path / "training.log"))

    logger.info(f"JAX {jax.__version__} | Flax {flax.__version__} | Optax {optax.__version__} | Orbax {ocp.__version__}")
    logger.info(f"devices: {jax.devices()}")
    logger.info(f"Train file: {train_file} | Output: {checkpoint_path} | Iterations: {iterations}")

    # snapshot the exact training script alongside the checkpoints
    script_path = Path(__file__).resolve()
    shutil.copy2(script_path, checkpoint_path / script_path.name)

    data_loader = FileSource(train_file, LOGISTIC_SCALING)

    batch_size = 1 << HYPERPARAM_BATCH_SIZE_LOG
    batch_size_adjustment = 14 - HYPERPARAM_BATCH_SIZE_LOG
    checkpoint_freq = 1 << (16 + batch_size_adjustment)
    validation_freq = 1 << (12 + batch_size_adjustment)
    steps_per_epoch = (len(data_loader) - 1) // batch_size + 1

    model = ZeuxoModel(rngs=nnx.Rngs(SEED))

    warmup_steps = iterations // 20  # 5%
    end_steps = iterations // 20  # 5%
    schedule = optax.join_schedules([
        optax.linear_schedule(0, HYPERPARAM_LEARNING_RATE_PEAK, warmup_steps),
        optax.cosine_decay_schedule(HYPERPARAM_LEARNING_RATE_PEAK, iterations - warmup_steps - end_steps, HYPERPARAM_LEARNING_RATE_END),
        optax.constant_schedule(HYPERPARAM_LEARNING_RATE_END),
    ], boundaries=[warmup_steps, iterations - end_steps])
    optimizer = nnx.Optimizer(model, optax.adamw(schedule))

    train_metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    test_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        winning_ratio=nnx.metrics.Average("winning_ratio"),
        losing_ratio=nnx.metrics.Average("losing_ratio"),
    )

    metrics_history: dict[str, list[float]] = {}
    for m in train_metrics.compute():
        metrics_history[f"train_{m}"] = []
    for m in test_metrics.compute():
        metrics_history[f"test_{m}"] = []

    checkpointer = ocp.StandardCheckpointer()
    with open(checkpoint_path / "metrics.csv", "w") as f:
        f.write("step,epoch," + ",".join(metrics_history.keys()) + "\n")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(parent_run_id=PARENT_RUN_ID, run_id=RUN_ID, description=MLFLOW_RUN_DESCRIPTION):
        mlflow.log_params({
            "train_file": str(train_file),
            "iterations": iterations,
            "batch_size": batch_size,
            "batch_size_log": HYPERPARAM_BATCH_SIZE_LOG,
            "logistic_scaling": LOGISTIC_SCALING,
            "size_accumulator": HYPERPARAM_ACCUMULATOR_SIZE,
            "size_hidden": HYPERPARAM_HIDDEN_SIZE,
            "peak_lr": HYPERPARAM_LEARNING_RATE_PEAK,
            "end_lr": HYPERPARAM_LEARNING_RATE_END,
            "warmup_steps": warmup_steps,
            "training_samples": data_loader.samples("train"),
            "test_samples": data_loader.samples("test"),
        })
        mlflow.set_tag("device", jax.devices()[0].device_kind)
        mlflow.log_artifact(str(script_path), artifact_path="code")

        logger.info(f"Training for {iterations} iterations")
        for step, batch in enumerate(data_loader.batched(batch_size, dataset="train")):
            train_step(model, optimizer, train_metrics, batch)

            if step % validation_freq == 0 or step == iterations - 1:
                epoch = step / steps_per_epoch
                for m, v in train_metrics.compute().items():
                    metrics_history[f"train_{m}"].append(float(v))
                train_metrics.reset()
                for test_batch in data_loader.batched(batch_size, dataset="test", repeat=False):
                    eval_step(model, test_metrics, test_batch)
                for m, v in test_metrics.compute().items():
                    metrics_history[f"test_{m}"].append(float(v))
                test_metrics.reset()

                logger.info(f"step {step:>8} (epoch {epoch:7.3f}): "
                            f"train {metrics_history['train_loss'][-1]:.5f} | test {metrics_history['test_loss'][-1]:.5f}")
                with open(checkpoint_path / "metrics.csv", "a") as f:
                    f.write(f"{step},{epoch:.5f}," + ",".join(str(metrics_history[m][-1]) for m in metrics_history) + "\n")
                mlflow.log_metric("epoch", epoch, step=step)
                mlflow.log_metric("learning_rate", float(schedule(step)), step=step)
                for m in metrics_history:
                    mlflow.log_metric(m, metrics_history[m][-1], step=step)

            if step != 0 and step % checkpoint_freq == 0:
                logger.info(f"Checkpointing at step {step}")
                _, state = nnx.split(model)
                checkpointer.save(str(checkpoint_path / f"state_{step}"), state)

            if step >= iterations - 1:
                break

        logger.info("Training complete; running validation set")
        test_metrics.reset()
        data_loader.clear_in_memory()
        for val_batch in data_loader.batched(batch_size, dataset="validation", repeat=False):
            eval_step(model, test_metrics, val_batch)
        for m, v in test_metrics.compute().items():
            logger.info(f"validation {m}: {float(v):.5f}")
            mlflow.log_metric(f"validation_{m}", float(v), step=iterations)

        state = nnx.state(model)
        checkpointer.save(str(checkpoint_path / "state"), state)
        checkpointer.wait_until_finished()
        mlflow.log_artifact(str(checkpoint_path / "state"), artifact_path="model")
        mlflow.log_artifact(str(checkpoint_path / "metrics.csv"), artifact_path="metrics")
        mlflow.log_artifact(str(checkpoint_path / "training.log"), artifact_path="logs")

    logging.shutdown()


if __name__ == "__main__":
    main()
