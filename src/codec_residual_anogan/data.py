"""Generic feature-split loading for any replay-detection dataset."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FeatureSplit:
    name: str
    features: np.ndarray
    ids: list
    labels: object = None


def split_paths(feature_dir, split):
    root = Path(feature_dir)
    return {
        "features": root / ("%s_residual.npy" % split),
        "ids": root / ("%s_ids.txt" % split),
        "labels": root / ("%s_labels.txt" % split),
    }


def read_ids(path):
    with Path(path).open() as handle:
        values = [line.strip() for line in handle if line.strip()]
    if len(values) != len(set(values)):
        raise RuntimeError("duplicate IDs in %s" % path)
    return values


def read_label_map(path):
    labels = {}
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(
                    "%s:%d requires ID and label" % (path, line_number))
            labels[fields[0]] = int(fields[1])
    return labels


def _all_finite(features, rows_per_chunk=8192):
    """Validate memmapped matrices without allocating a full-size mask."""
    return all(
        np.isfinite(features[start:start + rows_per_chunk]).all()
        for start in range(0, len(features), rows_per_chunk))


def load_split(feature_dir, split, require_labels=False, mmap=False):
    paths = split_paths(feature_dir, split)
    mode = "r" if mmap else None
    features = np.load(paths["features"], mmap_mode=mode)
    ids = read_ids(paths["ids"])
    if features.ndim != 2:
        raise RuntimeError(
            "%s must be a two-dimensional matrix" % paths["features"])
    if len(features) != len(ids):
        raise RuntimeError(
            "%s feature/ID count mismatch" % split)
    if not _all_finite(features):
        raise RuntimeError("%s contains non-finite features" % split)
    labels = None
    if paths["labels"].is_file():
        label_map = read_label_map(paths["labels"])
        missing = [sample_id for sample_id in ids if sample_id not in label_map]
        if missing:
            raise RuntimeError(
                "%s has missing labels; first=%s" % (split, missing[0]))
        labels = np.asarray([label_map[value] for value in ids], dtype=np.int8)
    elif require_labels:
        raise FileNotFoundError(paths["labels"])
    return FeatureSplit(
        split, features.astype(np.float32, copy=False), ids, labels)


def load_training_splits(feature_dir, train_splits, validation_split,
                         normal_label=1):
    train_features = []
    train_ids = []
    for split_name in train_splits:
        split = load_split(feature_dir, split_name)
        if split.labels is None:
            selected = np.arange(len(split.features))
        else:
            selected = np.flatnonzero(split.labels == int(normal_label))
        if not len(selected):
            raise RuntimeError("%s has no normal training rows" % split_name)
        train_features.append(split.features[selected])
        train_ids.extend([split.ids[index] for index in selected])
    if len(train_ids) != len(set(train_ids)):
        raise RuntimeError("duplicate IDs across training splits")

    validation = load_split(
        feature_dir, validation_split, require_labels=True)
    if set(validation.labels.tolist()) != {0, 1}:
        raise RuntimeError("validation split must contain labels 0 and 1")
    overlap = set(train_ids).intersection(validation.ids)
    if overlap:
        raise RuntimeError(
            "training/validation ID overlap; first=%s" % sorted(overlap)[0])
    return (
        np.ascontiguousarray(np.concatenate(train_features), dtype=np.float32),
        train_ids,
        validation,
    )
