from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, InputLayer, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_GROUPS = 8

# These 8 coarse groups match the post-processing logic in sign_engine.py
# where model outputs (0..7) are refined into final letters.
GROUP_TO_LETTERS = {
    0: ["A", "E", "M", "N", "S", "T"],
    1: ["B", "D", "F", "I", "K", "R", "U", "V", "W"],
    2: ["C", "O"],
    3: ["G", "H"],
    4: ["L"],
    5: ["P", "Q", "Z"],
    6: ["X"],
    7: ["J", "Y"],
}

LETTER_TO_GROUP = {
    letter: group_id
    for group_id, letters in GROUP_TO_LETTERS.items()
    for letter in letters
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the sign-to-text CNN (8-group classifier) used by sign_engine.py."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("AtoZ_3.1"),
        help=(
            "Dataset root folder. Supports either A-Z subfolders or direct group folders (0..7)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cnn8grps_rad1_model_trained.h5"),
        help="Output .h5 path for trained model.",
    )
    parser.add_argument("--image-size", type=int, default=400, help="Input image size.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (0.0 to <1.0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--rescale",
        action="store_true",
        help=(
            "If set, divide pixels by 255.0. Keep OFF to match current inference flow in sign_engine.py."
        ),
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file for custom folder/letter to group mapping. "
            "Example: {\"A\": 0, \"B\": 1, \"my_folder\": 3}"
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_custom_mapping(mapping_json: Path | None) -> Dict[str, int]:
    if mapping_json is None:
        return {}
    if not mapping_json.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_json}")

    raw = json.loads(mapping_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("mapping-json must be a JSON object (dictionary).")

    cleaned: Dict[str, int] = {}
    for key, value in raw.items():
        key_norm = str(key).strip().upper()
        group_id = int(value)
        if group_id < 0 or group_id >= NUM_GROUPS:
            raise ValueError(
                f"Invalid group id {group_id} for '{key}'. Must be 0..{NUM_GROUPS - 1}."
            )
        cleaned[key_norm] = group_id
    return cleaned


def normalize_group_folder_name(name: str) -> int | None:
    s = name.strip()
    if not s:
        return None

    low = s.lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "g0": 0,
        "g1": 1,
        "g2": 2,
        "g3": 3,
        "g4": 4,
        "g5": 5,
        "g6": 6,
        "g7": 7,
        "grp0": 0,
        "grp1": 1,
        "grp2": 2,
        "grp3": 3,
        "grp4": 4,
        "grp5": 5,
        "grp6": 6,
        "grp7": 7,
        "group0": 0,
        "group1": 1,
        "group2": 2,
        "group3": 3,
        "group4": 4,
        "group5": 5,
        "group6": 6,
        "group7": 7,
    }
    return aliases.get(low)


def resolve_group_id(folder_name: str, custom_mapping: Dict[str, int]) -> int | None:
    upper_name = folder_name.strip().upper()

    if upper_name in custom_mapping:
        return custom_mapping[upper_name]

    group_id = normalize_group_folder_name(folder_name)
    if group_id is not None:
        return group_id

    if len(upper_name) == 1 and "A" <= upper_name <= "Z":
        if upper_name in custom_mapping:
            return custom_mapping[upper_name]
        return LETTER_TO_GROUP.get(upper_name)

    return None


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_dataset_index(
    dataset_dir: Path,
    custom_mapping: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, Counter, Counter]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    path_data = []
    y_data = []
    folder_counts: Counter = Counter()
    group_counts: Counter = Counter()
    unresolved_folders = []

    folders = sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not folders:
        raise ValueError(f"No class folders found in: {dataset_dir}")

    for folder in folders:
        group_id = resolve_group_id(folder.name, custom_mapping)
        if group_id is None:
            unresolved_folders.append(folder.name)
            continue

        for image_path in iter_images(folder):
            path_data.append(str(image_path))
            y_data.append(group_id)
            folder_counts[folder.name] += 1
            group_counts[group_id] += 1

    if unresolved_folders:
        names = ", ".join(unresolved_folders)
        raise ValueError(
            "Could not map these folder names to the 8 groups. "
            f"Rename folders or pass --mapping-json: {names}"
        )

    if not path_data:
        raise ValueError("No valid images were loaded from dataset.")

    x = np.array(path_data, dtype=np.str_)
    y = np.array(y_data, dtype=np.int32)
    return x, y, folder_counts, group_counts


def _decode_preprocess(
    image_path: tf.Tensor,
    label: tf.Tensor,
    image_size: int,
    rescale: bool,
) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    image = tf.cast(image, tf.float32)
    if rescale:
        image = image / 255.0

    one_hot = tf.one_hot(label, depth=NUM_GROUPS, dtype=tf.float32)
    image = tf.ensure_shape(image, (image_size, image_size, 3))
    one_hot = tf.ensure_shape(one_hot, (NUM_GROUPS,))
    return image, one_hot


def make_tf_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    rescale: bool,
    seed: int,
    training: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training and len(paths) > 1:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 10000),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        lambda p, l: _decode_preprocess(p, l, image_size=image_size, rescale=rescale),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.ignore_errors()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def stratified_split(
    y: np.ndarray, val_split: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if val_split <= 0:
        all_idx = np.arange(len(y), dtype=np.int32)
        return all_idx, np.array([], dtype=np.int32)
    if val_split >= 1:
        raise ValueError("--val-split must be < 1.0")

    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []

    for cls in sorted(np.unique(y).tolist()):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        n_val = int(round(len(cls_idx) * val_split))
        if len(cls_idx) > 1:
            n_val = max(1, min(n_val, len(cls_idx) - 1))
        else:
            n_val = 0

        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx, dtype=np.int32), np.array(val_idx, dtype=np.int32)


def build_model(image_size: int) -> Sequential:
    model = Sequential(
        [
            InputLayer(input_shape=(image_size, image_size, 3)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(96, activation="relu"),
            Dropout(0.4),
            Dense(64, activation="relu"),
            Dense(NUM_GROUPS, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    custom_mapping = load_custom_mapping(args.mapping_json)
    x_paths, y, folder_counts, group_counts = build_dataset_index(
        dataset_dir=args.dataset_dir,
        custom_mapping=custom_mapping,
    )

    train_idx, val_idx = stratified_split(y=y, val_split=args.val_split, seed=args.seed)
    x_train_paths = x_paths[train_idx]
    y_train = y[train_idx]

    x_val_paths = x_paths[val_idx] if len(val_idx) else None
    y_val = y[val_idx] if len(val_idx) else None

    train_ds = make_tf_dataset(
        paths=x_train_paths,
        labels=y_train,
        image_size=args.image_size,
        batch_size=args.batch_size,
        rescale=args.rescale,
        seed=args.seed,
        training=True,
    )

    val_ds = None
    if x_val_paths is not None and y_val is not None and len(y_val):
        val_ds = make_tf_dataset(
            paths=x_val_paths,
            labels=y_val,
            image_size=args.image_size,
            batch_size=args.batch_size,
            rescale=args.rescale,
            seed=args.seed,
            training=False,
        )

    print(f"Dataset directory: {args.dataset_dir.resolve()}")
    print(f"Total samples: {len(y)}")
    print(f"Train samples: {len(y_train)}")
    print(f"Validation samples: {0 if y_val is None else len(y_val)}")
    print(f"Rescale enabled: {args.rescale}")
    print("\nFolder sample counts:")
    for folder_name in sorted(folder_counts.keys()):
        print(f"  {folder_name}: {folder_counts[folder_name]}")
    print("\nGroup sample counts:")
    for group_id in range(NUM_GROUPS):
        print(f"  {group_id}: {group_counts[group_id]}")

    model = build_model(args.image_size)
    model.summary()

    monitor = "val_accuracy" if val_ds is not None else "accuracy"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=args.patience,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"\nSaved model: {args.output.resolve()}")

    train_eval_ds = make_tf_dataset(
        paths=x_train_paths,
        labels=y_train,
        image_size=args.image_size,
        batch_size=args.batch_size,
        rescale=args.rescale,
        seed=args.seed,
        training=False,
    )
    train_metrics = model.evaluate(train_eval_ds, verbose=0)
    print(f"Train loss: {train_metrics[0]:.4f} | Train acc: {train_metrics[1]:.4f}")

    val_metrics = None
    if val_ds is not None:
        val_metrics = model.evaluate(val_ds, verbose=0)
        print(f"Val loss: {val_metrics[0]:.4f} | Val acc: {val_metrics[1]:.4f}")

    metadata = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "output_model": str(args.output.resolve()),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_split": args.val_split,
        "seed": args.seed,
        "rescale": bool(args.rescale),
        "num_groups": NUM_GROUPS,
        "group_to_letters": GROUP_TO_LETTERS,
        "samples_total": int(len(y)),
        "samples_train": int(len(y_train)),
        "samples_val": int(0 if y_val is None else len(y_val)),
        "folder_counts": dict(folder_counts),
        "group_counts": {str(k): int(v) for k, v in group_counts.items()},
        "history_keys": list(history.history.keys()),
        "final_train_metrics": {
            "loss": float(train_metrics[0]),
            "accuracy": float(train_metrics[1]),
        },
        "final_val_metrics": (
            None
            if val_metrics is None
            else {"loss": float(val_metrics[0]), "accuracy": float(val_metrics[1])}
        ),
        "custom_mapping_json": (
            None if args.mapping_json is None else str(args.mapping_json.resolve())
        ),
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata: {meta_path.resolve()}")


if __name__ == "__main__":
    main()
