# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "jax[cuda12]~=0.11.0",
#    "flax~=0.12.8",
#    "grain==0.2.18",
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
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Literal, cast

import jax
from jax import numpy as jnp
import grain
import numpy as np
from scipy.special import expit
import flax
from flax import nnx
import optax
import orbax.checkpoint as ocp
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import mlflow
import absl.flags

logger = logging.getLogger(__name__)

# --- board / feature geometry -------------------------------------------------
N_SQUARES = 64
N_PIECE_TYPES = 16  # 0 empty, 1 ep-capturable pawn, then our/their pawn..king with separate castling-rook tokens (see data/Encoding.scala)
MODEL_LAYERS = 16 # How many transformer blocks to use;
MODEL_WIDTH = 512 
ATTENTION_WIDTH = 64 # Width of each attention head
N_HEADS = MODEL_WIDTH // ATTENTION_WIDTH # How many attention heads to use in the transformer
assert ATTENTION_WIDTH * N_HEADS == MODEL_WIDTH, "Attention width must divide model width evenly"

LOGISTIC_SCALING = 400.0  # centipawns -> win prob via sigmoid(eval / scaling)

EVAL_SAMPLES = 1 << 16  # fixed subsample of the test split, kept on device for periodic eval
EVAL_POOL_SAMPLES = 1 << 22  # test rows the eval sample is permuted out of (head alone would be sequential positions from the same games)
EVAL_SEED = 271828  # independent of SEED so the eval sample is identical across runs
MAX_TRAIN_SAMPLES = 300_000_000  # cap on train rows materialised in RAM
EVAL_FREQ_BOARDS = 1 << 21  # boards trained between in-loop evals (~1% eval overhead)
CHECKPOINT_FREQ_BOARDS = int(os.environ.get("CHECKPOINT_FREQ_BOARDS", 1 << 25))  # boards between intermediate checkpoints; short runs only get the final one

CLIP_GRAD_NORM = 1.0
ADAM_B2 = 0.99

ATTENTION_IMPL = cast(Literal["xla", "cudnn"], os.environ.get("ATTENTION_IMPL", "cudnn"))

HYPERPARAM_LEARNING_RATE_PEAK = float(os.environ.get("HYPERPARAM_LEARNING_RATE_PEAK", 4e-4))
HYPERPARAM_LEARNING_RATE_END = float(os.environ.get("HYPERPARAM_LEARNING_RATE_END", 2e-6))
HYPERPARAM_BATCH_SIZE_LOG = int(os.environ.get("HYPERPARAM_BATCH_SIZE_LOG", 10))  # 2^10 = 1024

TRAIN_BOARDS = os.environ.get("TRAIN_BOARDS", None)
SEED = int(os.environ.get("SEED", 314159))

PARENT_RUN_ID = os.environ.get("PARENT_RUN_ID", None)
RUN_ID = os.environ.get("RUN_ID", None)
MLFLOW_RUN_DESCRIPTION = os.environ.get("MLFLOW_RUN_DESCRIPTION")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "zeuxo")

TRAIN_FILE = os.environ.get("TRAIN_FILE")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH")

class Attention(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, c_dtype=jnp.bfloat16):
        self.qkv = nnx.Linear(MODEL_WIDTH, ATTENTION_WIDTH * N_HEADS * 3, use_bias=False, rngs=rngs, dtype=c_dtype)
        self.o = nnx.Linear(ATTENTION_WIDTH * N_HEADS, MODEL_WIDTH, use_bias=False, rngs=rngs, dtype=c_dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (batch, seq_len, MODEL_WIDTH)
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], N_HEADS, 3 * ATTENTION_WIDTH)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        attn = jax.nn.dot_product_attention(q, k, v, implementation=ATTENTION_IMPL)
        o = self.o(attn.reshape(x.shape[0], x.shape[1], N_HEADS * ATTENTION_WIDTH))
        return o # (batch, seq_len, MODEL_WIDTH)


class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, c_dtype=jnp.bfloat16):
        self.fc1 = nnx.Linear(MODEL_WIDTH, MODEL_WIDTH * 4, use_bias=False, rngs=rngs, dtype=c_dtype)
        self.fc2 = nnx.Linear(MODEL_WIDTH * 4, MODEL_WIDTH, use_bias=False, rngs=rngs, dtype=c_dtype)
        
    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (batch, seq_len, MODEL_WIDTH)
        x = nnx.gelu(self.fc1(x))
        x = self.fc2(x)
        return x # (batch, seq_len, MODEL_WIDTH)

class TransformerBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, c_dtype=jnp.bfloat16, n_dtype=jnp.float32):
        self.attn = Attention(rngs=rngs, c_dtype=c_dtype)
        self.mlp = MLP(rngs=rngs, c_dtype=c_dtype)
        self.norm1 = nnx.RMSNorm(MODEL_WIDTH, rngs=rngs, dtype=n_dtype)
        self.norm2 = nnx.RMSNorm(MODEL_WIDTH, rngs=rngs, dtype=n_dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (batch, seq_len, MODEL_WIDTH)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x # (batch, seq_len, MODEL_WIDTH)
    

class ZeuxoModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.piece_embedding = nnx.Embed(N_PIECE_TYPES, MODEL_WIDTH, rngs=rngs)
        self.positional_embedding = nnx.Embed(N_SQUARES, MODEL_WIDTH, rngs=rngs)

        self.blocks = nnx.List([
            TransformerBlock(rngs=rngs, c_dtype=jnp.bfloat16, n_dtype=jnp.float32) for _ in range(MODEL_LAYERS)
        ])
        self.final_norm = nnx.RMSNorm(MODEL_WIDTH, rngs=rngs, dtype=jnp.float32)
        self.head = nnx.Linear(MODEL_WIDTH, 1, use_bias=False, rngs=rngs, dtype=jnp.bfloat16)


    def __call__(self, tokens: jax.Array) -> jax.Array:
        assert tokens.ndim == 2 and tokens.shape[1] == N_SQUARES # (batch, 64)

        # pos = jnp.broadcast_to(jax.nn.one_hot(jnp.arange(N_SQUARES), N_SQUARES), (tokens.shape[0], N_SQUARES, N_SQUARES)) # (batch, 64, 64)
        x = self.piece_embedding(tokens) + self.positional_embedding(jnp.arange(N_SQUARES)) # (batch, 64, MODEL_WIDTH)

        for block in self.blocks:
            x = block(x) # (batch, 64, MODEL_WIDTH)

        x = self.final_norm(x) # (batch, 64, MODEL_WIDTH)
        x_reduced = jnp.mean(x, axis=1) # (batch, MODEL_WIDTH)
        logits = jnp.squeeze(self.head(x_reduced), axis=-1) # (batch,)
        return logits.astype(jnp.float32) # (batch,)


def normalise_eval(eval: jax.Array, logistic_scaling: float) -> jax.Array:
    return nnx.sigmoid(eval / logistic_scaling)


def loss_fn(logits: jax.Array, labels: jax.Array) -> jax.Array:
    # binary cross entropy between predicted and target win probabilities
    assert logits.shape == labels.shape
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))


@nnx.jit
def train_step(model: ZeuxoModel, optimizer: nnx.ModelAndOptimizer, metrics: nnx.MultiMetric, batch: dict[str, jax.Array]):
    def loss(model: ZeuxoModel):
        return loss_fn(model(batch["features"]), batch["label"])

    loss_val, grads = nnx.value_and_grad(loss)(model)
    optimizer.update(grads)
    gnorm = optax.global_norm(grads)
    metrics.update(loss=loss_val, grad_norm=gnorm, grad_clipped=(gnorm > CLIP_GRAD_NORM).astype(jnp.float32))


@nnx.jit
def eval_step(model: ZeuxoModel, metrics: nnx.MultiMetric, batch: dict[str, jax.Array]):
    logits = model(batch["features"])
    pred = nnx.sigmoid(logits)
    metrics.update(
        loss=loss_fn(logits, batch["label"]),
        winning_ratio=jnp.mean(pred > 0.8),
        losing_ratio=jnp.mean(pred < 0.2),
    )


def load_test_data(path: Path) -> tuple[dict[str, jax.Array], int]:
    # eval sample must stay bit-identical to the old FileSource one: same scan order, permutation and device layout
    schema = pa.schema([
        pa.field("features", pa.list_(pa.int8(), N_SQUARES)),
        pa.field("eval", pa.int32()),
        pa.field("setType", pa.string()),
    ])
    partition = pds.partitioning(pa.schema([pa.field("setType", pa.string(), nullable=False)]), flavor="hive")
    ds = pds.dataset(path, partitioning=partition, schema=schema)
    test_rows = ds.count_rows(filter=pds.field("setType") == "test")
    pool = ds.head(EVAL_POOL_SAMPLES, columns=["features", "eval"], filter=pds.field("setType") == "test")
    perm = np.random.default_rng(EVAL_SEED).permutation(pool.num_rows)
    sample = pool.take(perm[:EVAL_SAMPLES])
    data = jax.device_put({
        "features": jnp.from_dlpack(sample.column("features").combine_chunks().values).reshape(-1, N_SQUARES),
        "label": normalise_eval(jnp.from_dlpack(sample.column("eval").combine_chunks()), LOGISTIC_SCALING),
    }, device=jax.devices()[0])
    logger.info(f"Eval sample of {len(data['label']):,} test boards (pool {pool.num_rows:,} of {test_rows:,}) on device {data['features'].device}")
    return data, test_rows


def main() -> None:
    assert TRAIN_FILE is not None, "TRAIN_FILE environment variable is not set."
    assert CHECKPOINT_PATH is not None, "CHECKPOINT_PATH environment variable is not set."
    assert TRAIN_BOARDS is not None, "TRAIN_BOARDS environment variable is not set."

    train_file = Path(TRAIN_FILE)
    assert train_file.exists(), f"Train file {train_file} does not exist."
    train_boards = int(TRAIN_BOARDS)
    batch_size = 1 << HYPERPARAM_BATCH_SIZE_LOG
    iterations = train_boards // batch_size
    assert iterations > 0, f"Train boards {train_boards} must be at least the batch size {batch_size}."

    checkpoint_root = Path(CHECKPOINT_PATH)
    assert checkpoint_root.exists(), f"Checkpoint path {checkpoint_root} does not exist."
    checkpoint_path = (checkpoint_root / pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")).resolve()
    checkpoint_path.mkdir()

    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(checkpoint_path / "training.log"))

    logger.info(f"JAX {jax.__version__} | Flax {flax.__version__} | Optax {optax.__version__} | Orbax {ocp.__version__}")
    logger.info(f"devices: {jax.devices()}")
    logger.info(f"Train file: {train_file} | Output: {checkpoint_path} | Boards: {train_boards:,} -> {iterations:,} iterations of {batch_size}")

    # snapshot the exact training script alongside the checkpoints
    script_path = Path(__file__).resolve()
    shutil.copy2(script_path, checkpoint_path / script_path.name)

    training_samples = [ str(p.absolute()) for p in train_file.glob("setType=train/*.parquet") ]
    random.Random(SEED).shuffle(training_samples)
    train_rows = pds.dataset(training_samples).count_rows()
    epochs = math.ceil(train_boards / train_rows)
    dataset = (grain.experimental.WindowShuffleIterDataset(
        grain.experimental.ParquetIterDataset(training_samples * epochs),
        window_size=1<<23,
        seed = SEED
        )
        .batch(batch_size, drop_remainder=True)
        .map(lambda batch: {
            "features": batch["features"],
            "label": expit(batch["eval"].astype(np.float32) / LOGISTIC_SCALING),
        })
        .mp_prefetch(grain.MultiprocessingOptions(num_workers=4))
    )
    
    logger.info(f"Train pipeline: {len(training_samples)} files, {train_rows:,} rows, {epochs} epoch(s)")

    test_data, test_rows = load_test_data(train_file)

    def test_batches():
        for i in range(0, len(test_data["label"]) - batch_size + 1, batch_size):
            yield {k: v[i:i + batch_size] for k, v in test_data.items()}

    validation_freq = EVAL_FREQ_BOARDS // batch_size
    steps_per_epoch = (train_rows - 1) // batch_size + 1

    model = ZeuxoModel(rngs=nnx.Rngs(SEED))
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    logger.info(f"Model parameters: {n_params:,} (width {MODEL_WIDTH}, layers {MODEL_LAYERS}, heads {N_HEADS})")

    warmup_steps = iterations // 20  # 5%
    end_steps = iterations // 20  # 5%
    schedule = optax.join_schedules([
        optax.linear_schedule(0, HYPERPARAM_LEARNING_RATE_PEAK, warmup_steps),
        optax.cosine_decay_schedule(HYPERPARAM_LEARNING_RATE_PEAK, iterations - warmup_steps - end_steps, HYPERPARAM_LEARNING_RATE_END / HYPERPARAM_LEARNING_RATE_PEAK),
        optax.constant_schedule(HYPERPARAM_LEARNING_RATE_END),
    ], boundaries=[warmup_steps, iterations - end_steps])
    def weight_decay_mask(params):
        def is_kernel(path, _):
            return any(getattr(k, "key", None) == "kernel" or getattr(k, "name", None) == "kernel" for k in path)
        return jax.tree_util.tree_map_with_path(is_kernel, params)

    optimizer = nnx.ModelAndOptimizer(model, 
        optax.chain(
            optax.clip_by_global_norm(CLIP_GRAD_NORM),
            optax.adamw(schedule, mask=weight_decay_mask, b2=ADAM_B2)
            )
        )

    checkpointer = ocp.StandardCheckpointer()
    checkpoint_freq = CHECKPOINT_FREQ_BOARDS // batch_size
    checkpoint_dir = checkpoint_path / "checkpoints"
    checkpoint_dir.mkdir()

    def save_checkpoint(name: str):
        checkpointer.save(checkpoint_dir / name, nnx.state(model, nnx.Param))
        logger.info(f"Saved checkpoint {name}")

    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        grad_norm=nnx.metrics.Average("grad_norm"),
        grad_clipped=nnx.metrics.Average("grad_clipped"),
    )
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

    with open(checkpoint_path / "metrics.csv", "w") as f:
        f.write("step,epoch," + ",".join(metrics_history.keys()) + "\n")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(parent_run_id=PARENT_RUN_ID, run_id=RUN_ID, description=MLFLOW_RUN_DESCRIPTION):
        mlflow.log_params({
            "train_file": str(train_file),
            "n_params": n_params,
            "model_width": MODEL_WIDTH,
            "model_layers": MODEL_LAYERS,
            "n_heads": N_HEADS,
            "train_boards": train_boards,
            "iterations": iterations,
            "batch_size": batch_size,
            "batch_size_log": HYPERPARAM_BATCH_SIZE_LOG,
            "logistic_scaling": LOGISTIC_SCALING,
            "peak_lr": HYPERPARAM_LEARNING_RATE_PEAK,
            "end_lr": HYPERPARAM_LEARNING_RATE_END,
            "clip_grad_norm": CLIP_GRAD_NORM,
            "adam_b2": ADAM_B2,
            "attention_impl": ATTENTION_IMPL,
            "warmup_steps": warmup_steps,
            "checkpoint_freq": checkpoint_freq,
            "training_samples": train_rows,
            "test_samples": test_rows,
        })
        mlflow.set_tag("device", jax.devices()[0].device_kind)
        mlflow.log_artifact(str(script_path), artifact_path="code")

        logger.info(f"Training for {train_boards:,} boards ({iterations:,} iterations)")
        t_last = time.perf_counter()
        t_train_total = 0.0
        last_timed_step = -1
        for step, batch in enumerate(dataset):
            train_step(model, optimizer, train_metrics, batch)

            if step % validation_freq == 0 or step == iterations - 1:
                window = time.perf_counter() - t_last
                t_train_total += window
                step_time = window / (step - last_timed_step)
                last_timed_step = step
                epoch = step / steps_per_epoch
                for m, v in train_metrics.compute().items():
                    metrics_history[f"train_{m}"].append(float(v))
                train_metrics.reset()
                for test_batch in test_batches():
                    eval_step(model, test_metrics, test_batch)
                for m, v in test_metrics.compute().items():
                    metrics_history[f"test_{m}"].append(float(v))
                test_metrics.reset()

                logger.info(f"step {step:>8} (epoch {epoch:7.3f}): "
                            f"train {metrics_history['train_loss'][-1]:.5f} | test {metrics_history['test_loss'][-1]:.5f} | "
                            f"{batch_size / step_time:,.0f} boards/s")
                with open(checkpoint_path / "metrics.csv", "a") as f:
                    f.write(f"{step},{epoch:.5f}," + ",".join(str(metrics_history[m][-1]) for m in metrics_history) + "\n")
                mlflow.log_metric("epoch", epoch, step=step)
                mlflow.log_metric("boards", step * batch_size, step=step)
                mlflow.log_metric("learning_rate", float(cast(jax.Array, schedule(step))), step=step)
                mlflow.log_metric("boards_per_second", batch_size / step_time, step=step)
                for m in metrics_history:
                    mlflow.log_metric(m, metrics_history[m][-1], step=step)
                t_last = time.perf_counter()

            if (step + 1) % checkpoint_freq == 0 and step + 1 < iterations:
                t_ckpt = time.perf_counter()
                save_checkpoint(f"step_{step:08d}")
                t_last += time.perf_counter() - t_ckpt

            if step >= iterations - 1:
                break

        logger.info(f"Training complete; mean step time {t_train_total / iterations:.3f}s "
                    f"({batch_size / (t_train_total / iterations):,.0f} boards/s)")
        save_checkpoint("final")
        checkpointer.wait_until_finished()
        mlflow.log_artifacts(str(checkpoint_dir), artifact_path="checkpoints")

        mlflow.log_artifact(str(checkpoint_path / "metrics.csv"), artifact_path="metrics")
        mlflow.log_artifact(str(checkpoint_path / "training.log"), artifact_path="logs")

    logging.shutdown()


if __name__ == "__main__":
    absl.flags.FLAGS.mark_as_parsed()  # avoid absl flag parsing errors when running as a script
    main()
