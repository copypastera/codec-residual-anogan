"""Dataset-agnostic training and validation selection CLI."""
import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ..data import load_training_splits
from ..inference import anomaly_scores, invert_features
from ..metrics import compute_metrics
from ..models import build_models
from ..preprocessing import FeaturePreprocessor
from ..training import build_optimizers, train_epoch
from ..utils import (
    atomic_json, atomic_npz, atomic_torch_save, canonical_hash,
    configure_device, load_yaml, seed_everything, utc_now)


METRIC_FIELDS = [
    "epoch", "eer", "threshold", "auc", "generator_loss",
    "discriminator_loss", "generator_grad_norm",
    "discriminator_grad_norm", "generator_output_mean",
    "generator_output_variance", "real_feature_variance",
    "training_seconds", "inference_seconds", "checkpoint"]


def _validate(config):
    required = (
        "run", "data", "preprocessing", "model", "training",
        "inference", "anomaly_score", "runtime")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError("missing config sections: %s" % missing)
    if not config["data"].get("train_splits"):
        raise ValueError("data.train_splits must not be empty")
    if not config["data"].get("validation_split"):
        raise ValueError("data.validation_split is required")
    if config["model"].get("architecture") != "conv_baseline":
        raise ValueError("only the best conv_baseline is supported")
    if config["training"].get("gan_type", "bce") != "bce":
        raise ValueError("only the best BCE objective is supported")
    if int(config["training"].get("keep_top_k_checkpoints", 3)) <= 0:
        raise ValueError(
            "training.keep_top_k_checkpoints must be positive")


def _append_metric(path, row):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in METRIC_FIELDS})


def _checkpoint_payload(epoch, generator, discriminator, optimizers,
                        schedulers, config, preprocessor, metrics):
    return {
        "schema_version": 2,
        "kind": "codec_residual_anogan",
        "epoch": int(epoch),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizers[0].state_dict(),
        "optimizer_d": optimizers[1].state_dict(),
        "scheduler_g": schedulers[0].state_dict(),
        "scheduler_d": schedulers[1].state_dict(),
        "config": config,
        "config_sha256": canonical_hash(config),
        "preprocessing_metadata": preprocessor.metadata(),
        "metrics": metrics,
        "saved_utc": utc_now(),
    }


def run(config_path):
    config = load_yaml(config_path)
    _validate(config)
    seed = int(config["run"].get("seed", 42))
    seed_everything(seed)
    output_dir = Path(config["run"]["output_dir"]).resolve()
    if output_dir.exists():
        raise FileExistsError(
            "output directory exists; choose a new run.output_dir: %s" %
            output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    preprocessing_dir = output_dir / "preprocessing"
    checkpoints_dir.mkdir(parents=True)
    preprocessing_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    feature_dir = Path(config["data"]["feature_directory"]).resolve()
    normal_label = int(config["data"].get("normal_label", 1))
    train_raw, train_ids, validation = load_training_splits(
        feature_dir,
        config["data"]["train_splits"],
        config["data"]["validation_split"],
        normal_label=normal_label)
    preprocessor = FeaturePreprocessor(
        kind=config["preprocessing"].get("type", "standard"),
        pca=config["preprocessing"].get("pca_dim"))
    preprocessor.fit(train_raw, train_ids)
    preprocessor.save(preprocessing_dir)
    train_features = preprocessor.transform(train_raw)
    validation_features = preprocessor.transform(validation.features)

    device = configure_device(
        config["runtime"].get("device", "cpu"),
        config["runtime"].get("memory_fraction", 0.0))
    torch.backends.cudnn.deterministic = bool(
        config["runtime"].get("deterministic_inference", False))
    generator, discriminator = build_models(
        preprocessor.output_dim, config["model"], device)
    optimizers, schedulers = build_optimizers(
        generator, discriminator, config)

    history = []
    best = None
    patience = int(config["training"].get("early_stopping_patience", 15))
    keep_top_k = int(config["training"].get("keep_top_k_checkpoints", 3))
    total_epochs = int(config["training"].get("epochs", 100))
    for epoch in range(1, total_epochs + 1):
        started = time.time()
        train_metrics = train_epoch(
            generator, discriminator, train_features, optimizers,
            config, device, epoch, seed)
        training_seconds = time.time() - started
        schedulers[0].step()
        schedulers[1].step()

        def update(batch, total, seen, elapsed):
            if batch == 1 or batch == total or batch % 20 == 0:
                print(
                    "validation batch=%d/%d seen=%d elapsed=%.1fs" %
                    (batch, total, seen, elapsed), flush=True)

        components, inference_seconds = invert_features(
            generator, discriminator, validation_features,
            config, device, seed * 1000003 + epoch * 17 + 1,
            status_callback=update)
        scores = anomaly_scores(components, config)
        metrics = compute_metrics(
            scores, validation.labels, normal_label=normal_label)
        checkpoint_path = checkpoints_dir / (
            "epoch_%03d.pt" % epoch)
        row = {
            "epoch": epoch,
            **metrics,
            **train_metrics,
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "checkpoint": str(checkpoint_path),
        }
        atomic_torch_save(
            checkpoint_path,
            _checkpoint_payload(
                epoch, generator, discriminator, optimizers,
                schedulers, config, preprocessor, row))
        _append_metric(output_dir / "metrics.csv", row)
        history.append(row)
        history.sort(key=lambda value: (value["eer"], value["epoch"]))
        while len(history) > keep_top_k:
            removed = history.pop()
            Path(removed["checkpoint"]).unlink(missing_ok=True)

        if best is None or (row["eer"], epoch) < (
                best["eer"], best["epoch"]):
            best = row
            released_checkpoint = output_dir / "best_checkpoint.pt"
            shutil.copy2(checkpoint_path, released_checkpoint)
            atomic_npz(
                output_dir / "best_validation_scores.npz",
                sample_id=np.asarray(validation.ids),
                label=validation.labels,
                anomaly_score=scores)
            atomic_json(output_dir / "selection.json", {
                "selection_split": validation.name,
                "best_epoch": epoch,
                "best_eer": row["eer"],
                "best_threshold": row["threshold"],
                "best_checkpoint": str(released_checkpoint),
                "updated_utc": utc_now(),
            })

        print(
            "epoch=%03d g_loss=%.6f d_loss=%.6f "
            "validation_eer=%.4f%% threshold=%.8f" %
            (epoch, train_metrics["generator_loss"],
             train_metrics["discriminator_loss"],
             100.0 * metrics["eer"], metrics["threshold"]),
            flush=True)
        if patience and epoch - int(best["epoch"]) >= patience:
            print("early stopping after %d non-improving epochs" % patience)
            break
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the best codec-residual AnoGAN on generic splits")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    return run(parse_args().config)


if __name__ == "__main__":
    raise SystemExit(main())
