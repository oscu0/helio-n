import json

import numpy as np
import os

import PIL

from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = str(MODULE_DIR.parent) + "/"

os.environ["KERAS_BACKEND"] = "jax"
import keras

# keras.config.set_backend("jax")
from keras import layers, ops

from Library import IO
from Library.Config import paths


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
    # decode paths if coming in as bytes
    if isinstance(fits_path, (bytes, np.bytes_)):
        fits_path = fits_path.decode("utf-8")
    if isinstance(mask_path, (bytes, np.bytes_)):
        mask_path = mask_path.decode("utf-8")

    # load 2-D arrays
    img = np.asarray(IO.prepare_fits(fits_path), dtype=np.float32)
    mask = np.asarray(IO.prepare_mask(mask_path), dtype=np.float32)

    # resize
    img_resized = PIL.Image.fromarray((img * 255).astype(np.uint8)).resize(
        (model_params["img_size"], model_params["img_size"]),
        resample=PIL.Image.BILINEAR,
    )
    mask_resized = PIL.Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (model_params["img_size"], model_params["img_size"]), resample=PIL.Image.NEAREST
    )

    # normalize back to [0,1] and add channel axis
    img = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=-1)
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
                img, mask = load_pair(fits_paths[j], mask_paths[j])
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
    inputs = keras.Input(shape=input_shape)

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

    model = keras.Model(inputs, outputs, name="CH_UNet_5lvl")
    return model


def dice_coef(y_true, y_pred, smooth=1.0):
    # flatten per-sample
    y_true_f = ops.reshape(y_true, (ops.shape(y_true)[0], -1))
    y_pred_f = ops.reshape(y_pred, (ops.shape(y_pred)[0], -1))

    intersection = ops.sum(y_true_f * y_pred_f, axis=1)
    denom = ops.sum(y_true_f, axis=1) + ops.sum(y_pred_f, axis=1)

    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return ops.mean(dice)


def bce_dice_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1.0 - dice_coef(y_true, y_pred)
    return 0.4 * ops.mean(bce) + 0.6 * dice_loss


def train_model(pairs_df, model_params, keep_every=3, path=None):
    fits_paths = pairs_df["fits_path"].astype(str).tolist()
    mask_paths = pairs_df["mask_path"].astype(str).tolist()

    if len(fits_paths) == 0:
        raise RuntimeError("pairs_df is empty: no FITS-mask pairs to train on.")
    if len(fits_paths) != len(mask_paths):
        raise RuntimeError(
            f"Mismatch in pairs_df: {len(fits_paths)} FITS vs {len(mask_paths)} masks."
        )

    n_total = len(fits_paths)
    print(f"Training on {n_total} FITS-mask pairs")

    # 90/10 split
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val

    train_fits = fits_paths[:n_train]
    train_masks = mask_paths[:n_train]
    val_fits = fits_paths[n_train:]
    val_masks = mask_paths[n_train:]

    # --- use every Nth training sample ---
    train_fits = train_fits[::keep_every]
    train_masks = train_masks[::keep_every]

    # steps per epoch (integer)
    steps_per_epoch = max(1, n_train // model_params["batch_size"])
    val_steps = max(1, n_val // model_params["batch_size"])

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

    model = build_unet(
        model_params=model_params
    )  # must be built with keras.layers, not tf.keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=model_params["learning_rate"]),
        loss=bce_dice_loss,  # your keras.ops-based loss
        metrics=[dice_coef, "accuracy"],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            paths["model_path"],
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        early_stop,
    ]

    model.fit(
        train_gen,
        epochs=model_params["num_epochs"],
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    if path is None:
        return model
    else:
        model.save(path)
        return path


def load_trained_model(architecture, date_range):
    architecture_json = json.load(
        open(PROJECT_ROOT + "Config/Model/Architecture/" + architecture + ".json")
    )
    date_range_json = json.load(
        open(PROJECT_ROOT + "Config/Model/Date Range/" + date_range + ".json")
    )
    path = PROJECT_ROOT + "Outputs/Models/" + architecture + date_range + ".keras"
    print(path, architecture_json, date_range_json)

    custom_objects = {
        "bce_dice_loss": bce_dice_loss,
        "dice_coef": dice_coef,
    }
    
    model = HelioNModel(
        keras.models.load_model(path, custom_objects=custom_objects),
        architecture,
        date_range
    )

    return model


class HelioNModel:
    def __init__(
        self,
        model,
        architecture_id: str,
        date_range_id: str,
        *,
        arch_dir="Config/Model/Architecture",
        date_dir="Config/Model/Date Range",
        root=".",
    ):
        self.model = model

        self.architecture_id = architecture_id
        self.date_range_id = date_range_id

        root = Path(root).resolve()

        self.architecture_path = root / arch_dir / f"{architecture_id}.json"
        self.date_range_path = root / date_dir / f"{date_range_id}.json"

        self.architecture = self._read_json(self.architecture_path)
        self.date_range = self._read_json(self.date_range_path)

    @staticmethod
    def _read_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def __str__(self):
        return "Wrapped model " + self.architecture_id + self.date_range_id
    
    def __repr__(self):
        return self.__str__()