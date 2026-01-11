import os
import time
import platform
from pathlib import Path

import numpy as np
import PIL

# TensorFlow C++ logging: WARNING+ERROR only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# absl / XLA logging: WARNING+ERROR only
os.environ["ABSL_LOG_LEVEL"] = "2"
import tensorflow as tf

from Library import IO
from Library.Config import (
    paths,
    train_max_queue_size,
    train_use_multiprocessing,
    train_workers,
)
from Models import load_architecture, load_date_range

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = str(MODULE_DIR.parent) + "/"

# Shortcuts to tf.keras namespaces
layers = tf.keras.layers


def augment_pair(img, mask):
    # --- random horizontal flip ---
    if np.random.rand() < 0.5:
        img = img[:, ::-1, :]
        mask = mask[:, ::-1, :]

    # --- small random brightness scaling ---
    scale = np.random.uniform(0.9, 1.1)  # ±10 %
    img = img * scale
    img = np.clip(img, 0.0, 1.0)

    return img, mask


def load_pair(fits_path, mask_path, model_params):
    """Deterministic preprocessing for one FITS/mask pair (no augmentation)."""
    if isinstance(fits_path, (bytes, np.bytes_)):
        fits_path = fits_path.decode("utf-8")
    if isinstance(mask_path, (bytes, np.bytes_)):
        mask_path = mask_path.decode("utf-8")

    _, img = IO.prepare_fits(fits_path)
    img = np.asarray(img, dtype=np.float32)
    mask = np.asarray(IO.prepare_mask(mask_path), dtype=np.float32)

    if model_params["avoid_requantization"]:
        # resize without quantizing the normalized FITS to 8-bit
        img_resized = IO.resize_for_model(img, model_params["img_size"])
    else:
        img_u8 = (img * 255).astype(np.uint8)
        img_resized = (
            PIL.Image.fromarray(img_u8)
            .resize(
                (model_params["img_size"], model_params["img_size"]),
                resample=PIL.Image.BILINEAR,
            )
            .convert("F")
        )
        img_resized = np.array(img_resized, dtype=np.float32) / 255.0

    mask_resized = PIL.Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (model_params["img_size"], model_params["img_size"]),
        resample=PIL.Image.NEAREST,
    )

    img = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=-1)
    mask = np.expand_dims(np.array(mask_resized, dtype=np.float32) / 255.0, axis=-1)
    mask = (mask > 0.5).astype(np.float32)
    return img, mask


def pair_generator(fits_paths, mask_paths, model_params, augment=False):
    n = len(fits_paths)
    idxs = np.arange(n)

    while True:
        np.random.shuffle(idxs)
        for i in range(0, n, model_params["batch_size"]):
            batch_idx = idxs[i : i + model_params["batch_size"]]
            imgs, masks = [], []
            for j in batch_idx:
                img, mask = load_pair(fits_paths[j], mask_paths[j], model_params)
                if augment:
                    img, mask = augment_pair(img, mask)

                imgs.append(img)
                masks.append(mask)

            if not imgs:
                continue

            X = np.stack(imgs, axis=0).astype(np.float32)  # (B, H, W, 1)
            Y = np.stack(masks, axis=0).astype(np.float32)  # (B, H, W, 1)
            yield X, Y


def double_conv(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_unet(
    model_params,
):
    input_shape = (model_params["img_size"], model_params["img_size"], 1)
    base_filters = model_params["base_filters"]
    inputs = tf.keras.Input(shape=input_shape)

    # ----- Encoder -----
    c1 = double_conv(inputs, base_filters)  # 256 x 256,  f
    p1 = layers.MaxPool2D(2)(c1)  # 128 x 128

    c2 = double_conv(p1, base_filters * 2)  # 128 x 128,  2f
    p2 = layers.MaxPool2D(2)(c2)  # 64 x 64

    c3 = double_conv(p2, base_filters * 4)  # 64 x 64,    4f
    p3 = layers.MaxPool2D(2)(c3)  # 32 x 32

    c4 = double_conv(p3, base_filters * 8)  # 32 x 32,    8f
    p4 = layers.MaxPool2D(2)(c4)  # 16 x 16

    # extra encoder level
    c5 = double_conv(p4, base_filters * 16)  # 16 x 16,   16f
    p5 = layers.MaxPool2D(2)(c5)  # 8 x 8

    # ----- Bottleneck -----
    bn = double_conv(
        p5, base_filters * 16
    )  # keep 16f; base*32 is also possible but heavier

    # ----- Decoder -----
    u5 = layers.Conv2DTranspose(base_filters * 16, 2, strides=2, padding="same")(bn)
    u5 = layers.Concatenate()([u5, c5])
    c6 = double_conv(u5, base_filters * 16)

    u4 = layers.Conv2DTranspose(base_filters * 8, 2, strides=2, padding="same")(c6)
    u4 = layers.Concatenate()([u4, c4])
    c7 = double_conv(u4, base_filters * 8)

    u3 = layers.Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same")(c7)
    u3 = layers.Concatenate()([u3, c3])
    c8 = double_conv(u3, base_filters * 4)

    u2 = layers.Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(c8)
    u2 = layers.Concatenate()([u2, c2])
    c9 = double_conv(u2, base_filters * 2)

    u1 = layers.Conv2DTranspose(base_filters, 2, strides=2, padding="same")(c9)
    u1 = layers.Concatenate()([u1, c1])
    c10 = double_conv(u1, base_filters)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c10)

    model = tf.keras.Model(inputs, outputs, name="CH_UNet_5lvl")
    return model


def dice_coef(y_true, y_pred, smooth=1.0):
    # flatten per-sample
    y_true_f = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
    y_pred_f = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)

    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return tf.reduce_mean(dice)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1.0 - dice_coef(y_true, y_pred)
    return 0.4 * tf.reduce_mean(bce) + 0.6 * dice_loss


def safe_fit(model, *args, **kwargs):
    try:
        return model.fit(*args, **kwargs)
    except TypeError as exc:
        msg = str(exc)
        if (
            "workers" not in msg
            and "use_multiprocessing" not in msg
            and "max_queue_size" not in msg
        ):
            raise
        for key in ("workers", "use_multiprocessing", "max_queue_size"):
            kwargs.pop(key, None)
        print("model.fit does not support multiprocessing args; retrying without them.")
        return model.fit(*args, **kwargs)


class TrainMonitor(tf.keras.callbacks.Callback):
    def __init__(
        self,
        *,
        overfit_loss_gap=0.1,
        overfit_dice_gap=0.1,
        underfit_dice=0.15,
    ):
        super().__init__()
        self.overfit_loss_gap = overfit_loss_gap
        self.overfit_dice_gap = overfit_dice_gap
        self.underfit_dice = underfit_dice
        self.train_start = None
        self.epoch_start = None

    @staticmethod
    def _fmt(value):
        if value is None:
            return "n/a"
        return f"{value:.4f}"

    def on_train_begin(self, logs=None):
        self.train_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        train_dice = logs.get("dice_coef")
        val_dice = logs.get("val_dice_coef")
        train_acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")

        notes = []
        if train_loss is not None and val_loss is not None:
            if val_loss - train_loss > self.overfit_loss_gap:
                notes.append("overfit-loss")
        if train_dice is not None and val_dice is not None:
            if train_dice - val_dice > self.overfit_dice_gap:
                notes.append("overfit-dice")
            if train_dice < self.underfit_dice and val_dice < self.underfit_dice:
                notes.append("underfit-dice")

        elapsed_total = time.time() - self.train_start if self.train_start else 0.0
        elapsed_epoch = time.time() - self.epoch_start if self.epoch_start else 0.0

        note_str = f" notes={','.join(notes)}" if notes else ""
        print(
            f"\n[monitor] epoch {epoch + 1} "
            f"elapsed_epoch={elapsed_epoch:.1f}s elapsed_total={elapsed_total:.1f}s "
            f"loss={self._fmt(train_loss)} val_loss={self._fmt(val_loss)} "
            f"dice={self._fmt(train_dice)} val_dice={self._fmt(val_dice)} "
            f"acc={self._fmt(train_acc)} val_acc={self._fmt(val_acc)}"
            f"{note_str}"
        )

    def on_train_end(self, logs=None):
        if self.train_start is None:
            return
        elapsed_total = time.time() - self.train_start
        print(f"\n[monitor] training elapsed_total={elapsed_total:.1f}s")


def train_model(pairs_df, model_params, keep_every=3, path=None, val_df=None):
    if val_df is None:
        raise ValueError("val_df is required; build the split in Models/<A?>.py.")

    train_df = pairs_df

    fits_paths = train_df["fits_path"].astype(str).tolist()
    mask_paths = train_df["mask_path"].astype(str).tolist()

    if len(fits_paths) == 0:
        raise RuntimeError("pairs_df is empty: no FITS-mask pairs to train on.")
    if len(fits_paths) != len(mask_paths):
        raise RuntimeError(
            f"Mismatch in pairs_df: {len(fits_paths)} FITS vs {len(mask_paths)} masks."
        )

    val_fits = val_df["fits_path"].astype(str).tolist()
    val_masks = val_df["mask_path"].astype(str).tolist()

    if len(val_fits) == 0:
        raise RuntimeError("val_df is empty: no FITS-mask pairs to validate on.")
    if len(val_fits) != len(val_masks):
        raise RuntimeError(
            f"Mismatch in val_df: {len(val_fits)} FITS vs {len(val_masks)} masks."
        )

    n_train = len(fits_paths)
    n_val = len(val_fits)
    n_total = n_train + n_val
    print(f"Training on {n_total} FITS-mask pairs ({n_train} train, {n_val} val)")

    train_fits = fits_paths
    train_masks = mask_paths

    # --- use every Nth training sample ---
    train_fits = train_fits[::keep_every]
    train_masks = train_masks[::keep_every]

    if model_params.get("correct_steps_by_n", False):
        n_train_eff = len(train_fits)
        n_val_eff = len(val_fits)
        steps_per_epoch = max(1, n_train_eff // model_params["batch_size"])
        val_steps = max(1, n_val_eff // model_params["batch_size"])
    else:
        steps_per_epoch = n_train
        val_steps = n_val

    train_gen = pair_generator(
        train_fits,
        train_masks,
        augment=True,
        model_params=model_params,
    )

    val_gen = pair_generator(
        val_fits,
        val_masks,
        augment=False,
        model_params=model_params,
    )

    model = build_unet(model_params=model_params)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=model_params["learning_rate"]),
        loss=bce_dice_loss,  # your keras.ops-based loss
        metrics=[dice_coef, "accuracy"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            path,
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
        early_stop,
        TrainMonitor(),
    ]

    safe_fit(
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

    if path is None:
        return model
    else:
        model.save(path)
        return path


def load_trained_model(architecture, date_range):
    path = PROJECT_ROOT + "Outputs/Models/" + architecture + date_range + ".keras"
    print(path)

    custom_objects = {
        "bce_dice_loss": bce_dice_loss,
        "dice_coef": dice_coef,
    }

    model = HelioNModel(
        tf.keras.models.load_model(path, custom_objects=custom_objects),
        architecture,
        date_range,
    )

    return model


class HelioNModel:
    def __init__(
        self,
        model,
        architecture_id: str,
        date_range_id: str,
    ):
        self.model = model

        self.architecture_id = architecture_id
        self.date_range_id = date_range_id

        self.architecture = load_architecture(architecture_id)
        self.date_range = load_date_range(architecture_id, date_range_id)

        # Lazily-built inference fn; keep None until first predict to avoid device issues
        self._infer = None
        self._prefer_jit = platform.system().lower().startswith("linux")

    def __str__(self):
        return "Wrapped model " + self.architecture_id + self.date_range_id

    def __repr__(self):
        return self.__str__()

    def compiled_infer(self, x):
        """Fast inference with a cached tf.function; falls back to direct call on failure."""
        x_tensor = tf.convert_to_tensor(x)

        if self._infer is None:
            self._infer = self._build_infer(self._prefer_jit)

        if self._infer is not None:
            try:
                return self._infer(x_tensor).numpy()
            except Exception:
                # If a jit-compiled fn fails, retry once without jit
                if self._prefer_jit:
                    self._infer = self._build_infer(False)
                    if self._infer is not None:
                        try:
                            return self._infer(x_tensor).numpy()
                        except Exception:
                            pass
                # Invalidate and fall back to eager call
                self._infer = None

        return self.model(x_tensor, training=False).numpy()

    # Backwards-compatible alias
    # def predict(self, x):
    #     return self.compiled_infer(x)

    def _build_infer(self, prefer_jit: bool):
        if prefer_jit:
            try:
                return tf.function(
                    lambda t: self.model(t, training=False),
                    jit_compile=True,
                    reduce_retracing=True,
                )
            except Exception:
                pass

        try:
            return tf.function(
                lambda t: self.model(t, training=False), reduce_retracing=True
            )
        except Exception:
            return None
