#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
import sys

sys.path.append(str(ROOT_DIR))

from Library import Model
from Library.Config import (
    paths,
    train_batch_size,
    train_max_queue_size,
    train_use_multiprocessing,
    train_workers,
)
from Models import load_architecture, load_date_range


def make_synthetic_pairs(n_samples: int, img_size: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:img_size, 0:img_size]

    cx = (img_size - 1) / 2.0
    cy = (img_size - 1) / 2.0
    r = img_size * 0.45
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2

    pairs = []
    for i in range(n_samples):
        img = rng.normal(loc=0.45, scale=0.08, size=(img_size, img_size)).astype(
            np.float32
        )
        img = np.clip(img, 0.0, 1.0)

        # Random ellipse mask inside the disk
        ecx = rng.uniform(img_size * 0.3, img_size * 0.7)
        ecy = rng.uniform(img_size * 0.3, img_size * 0.7)
        a = rng.uniform(img_size * 0.08, img_size * 0.18)
        b = rng.uniform(img_size * 0.08, img_size * 0.18)
        theta = rng.uniform(0, np.pi)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x0 = xx - ecx
        y0 = yy - ecy
        xr = x0 * cos_t + y0 * sin_t
        yr = -x0 * sin_t + y0 * cos_t
        ellipse = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0

        mask = np.zeros((img_size, img_size), dtype=np.float32)
        mask[ellipse] = 1.0

        # Correlate intensity with mask (darken inside ellipse)
        img[ellipse] *= rng.uniform(0.15, 0.35)

        # Zero outside disk
        img[~disk] = 0.0
        mask[~disk] = 0.0

        pairs.append((img[..., np.newaxis], mask[..., np.newaxis]))

    return pairs


def array_pair_generator(pairs, model_params, augment=False):
    n = len(pairs)
    idxs = np.arange(n)

    while True:
        np.random.shuffle(idxs)
        for i in range(0, n, model_params["batch_size"]):
            batch_idx = idxs[i : i + model_params["batch_size"]]
            imgs, masks = [], []
            for j in batch_idx:
                img, mask = pairs[j]
                if augment:
                    img, mask = Model.augment_pair(img.copy(), mask.copy())
                imgs.append(img)
                masks.append(mask)

            if not imgs:
                continue

            X = np.stack(imgs, axis=0).astype(np.float32)
            Y = np.stack(masks, axis=0).astype(np.float32)
            yield X, Y


def split_pairs(pairs, keep_every):
    n_total = len(pairs)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train = pairs[:n_train]
    val = pairs[n_train:]
    train = train[::keep_every]
    return train, val, n_train, n_val


def compute_steps(n_train, n_val, n_train_eff, n_val_eff, model_params):
    if model_params.get("correct_steps_by_n", False):
        steps_per_epoch = max(1, n_train_eff // model_params["batch_size"])
        val_steps = max(1, n_val_eff // model_params["batch_size"])
    else:
        steps_per_epoch = max(1, n_train // model_params["batch_size"])
        val_steps = max(1, n_val // model_params["batch_size"])
    return steps_per_epoch, val_steps


def train_pass(tag, train_gen, val_gen, steps_per_epoch, val_steps, model_params):
    model = Model.build_unet(model_params=model_params)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=model_params["learning_rate"]),
        loss=Model.bce_dice_loss,
        metrics=[Model.dice_coef, "accuracy"],
    )

    out_dir = ROOT_DIR / "Outputs" / "Models"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"Overfit_{tag}.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        Model.TrainMonitor(),
    ]

    history = Model.safe_fit(
        model,
        train_gen,
        epochs=model_params["num_epochs"],
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        workers=train_workers,
        use_multiprocessing=train_use_multiprocessing,
        max_queue_size=train_max_queue_size,
    )
    return model, history


def _stats(arr):
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def report_sample(tag, model, img, mask):
    pred_eval = model(img[np.newaxis, ...], training=False).numpy()[0, ..., 0]
    pred_train = model(img[np.newaxis, ...], training=True).numpy()[0, ..., 0]
    diff = pred_train - pred_eval

    mask_mean = float(np.mean(mask))
    print(
        f"[{tag}] pred eval stats: {_stats(pred_eval)} "
        f"pred train stats: {_stats(pred_train)} "
        f"diff stats: {_stats(diff)} "
        f"diff maxabs: {float(np.max(np.abs(diff))):.6f} "
        f"mask_mean: {mask_mean:.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Overfit test: synthetic vs real pairs.")
    parser.add_argument("architecture_id")
    parser.add_argument("date_range_id")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("samples must be >= 2 to keep a non-empty train split.")

    model_params = load_architecture(args.architecture_id)
    model_params.setdefault("batch_size", int(train_batch_size))
    date_range = load_date_range(args.architecture_id, args.date_range_id)
    keep_every = int(date_range.keep_every)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # --- Synthetic pass ---
    synth_pairs = make_synthetic_pairs(args.samples, model_params["img_size"], args.seed)
    synth_train, synth_val, n_train, n_val = split_pairs(synth_pairs, keep_every)
    synth_train_gen = array_pair_generator(
        synth_train, model_params, augment=True
    )
    synth_val_gen = array_pair_generator(
        synth_val, model_params, augment=False
    )
    steps_per_epoch, val_steps = compute_steps(
        n_train, n_val, len(synth_train), len(synth_val), model_params
    )
    print(
        "[synthetic] n_total="
        f"{len(synth_pairs)} n_train={len(synth_train)} n_val={len(synth_val)} "
        f"steps={steps_per_epoch} val_steps={val_steps}"
    )
    run_label = f"{args.architecture_id}{args.date_range_id}"
    model, history = train_pass(
        f"synthetic_{run_label}",
        synth_train_gen,
        synth_val_gen,
        steps_per_epoch,
        val_steps,
        model_params,
    )
    report_sample("synthetic", model, synth_train[0][0], synth_train[0][1])
    print(f"[synthetic] final metrics: { {k: v[-1] for k, v in history.history.items()} }")

    # --- Real FITS/mask pass ---
    df = pd.read_parquet(Path(paths["artifact_root"]) / "Paths.parquet")
    df = df.sort_index()
    real_df, _ = date_range.select_pairs(df)
    if real_df.empty:
        raise RuntimeError("No real FITS/mask pairs for the specified date range.")
    real_pairs = list(
        zip(real_df["fits_path"].astype(str), real_df["mask_path"].astype(str))
    )[: args.samples]
    real_train, real_val, n_train, n_val = split_pairs(real_pairs, keep_every)
    train_fits = [p[0] for p in real_train]
    train_masks = [p[1] for p in real_train]
    val_fits = [p[0] for p in real_val]
    val_masks = [p[1] for p in real_val]
    real_train_gen = Model.pair_generator(
        train_fits, train_masks, model_params=model_params, augment=True
    )
    real_val_gen = Model.pair_generator(
        val_fits, val_masks, model_params=model_params, augment=False
    )
    steps_per_epoch, val_steps = compute_steps(
        n_train, n_val, len(real_train), len(real_val), model_params
    )
    print(
        "[real] n_total="
        f"{len(real_pairs)} n_train={len(real_train)} n_val={len(real_val)} "
        f"steps={steps_per_epoch} val_steps={val_steps}"
    )
    model, history = train_pass(
        f"real_{run_label}",
        real_train_gen,
        real_val_gen,
        steps_per_epoch,
        val_steps,
        model_params,
    )
    img, mask = Model.load_pair(real_train[0][0], real_train[0][1], model_params)
    report_sample("real", model, img, mask)
    print(f"[real] final metrics: { {k: v[-1] for k, v in history.history.items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
