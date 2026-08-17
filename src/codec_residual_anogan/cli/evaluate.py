"""Evaluate a checkpoint on any generic feature split."""
import argparse
import math
import time
from pathlib import Path

import joblib
import numpy as np

from ..data import load_split
from ..inference import anomaly_scores, invert_features
from ..metrics import compute_metrics
from ..models import build_models
from ..utils import (
    atomic_json, atomic_npz, configure_device, sha256_file,
    torch_load, utc_now)


def _score_in_chunks(split, preprocessor, generator, discriminator, config,
                     device, seed, chunk_size):
    total = len(split.features)
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    chunk_count = int(math.ceil(total / float(chunk_size)))
    component_chunks = {}
    inversion_seconds = 0.0
    random_state = np.random.RandomState(seed)
    started = time.time()
    for chunk_index, start in enumerate(range(0, total, chunk_size), 1):
        stop = min(start + chunk_size, total)
        transformed = preprocessor.transform(split.features[start:stop])

        def update(batch, batches, seen, elapsed):
            if batch == 1 or batch == batches or batch % 20 == 0:
                print(
                    "chunk=%d/%d rows=%d:%d batch=%d/%d "
                    "elapsed=%.1fs" %
                    (chunk_index, chunk_count, start, stop, batch, batches,
                     time.time() - started), flush=True)

        values, elapsed = invert_features(
            generator, discriminator, transformed, config, device,
            seed=seed, status_callback=update, random_state=random_state)
        inversion_seconds += elapsed
        for name, array in values.items():
            component_chunks.setdefault(name, []).append(array)
    components = {
        name: np.concatenate(arrays)
        for name, arrays in component_chunks.items()}
    return components, inversion_seconds, time.time() - started


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = configure_device(args.device, args.memory_fraction)
    checkpoint = torch_load(args.checkpoint, device)
    config = checkpoint["config"]
    preprocessor = joblib.load(args.preprocessor)
    split = load_split(
        args.feature_dir, args.split, require_labels=False, mmap=True)
    generator, discriminator = build_models(
        preprocessor.output_dim, config["model"], device)
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])

    components, inversion_seconds, total_seconds = _score_in_chunks(
        split, preprocessor, generator, discriminator, config, device,
        args.seed, args.chunk_size)
    scores = anomaly_scores(components, config)
    arrays = {
        "sample_id": np.asarray(split.ids),
        "anomaly_score": scores,
        **components,
    }
    if split.labels is not None:
        arrays["label"] = split.labels
    atomic_npz(output_dir / "scores.npz", **arrays)

    result = {
        "kind": "codec_residual_anogan_evaluation",
        "split": split.name,
        "samples": len(split.ids),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "preprocessor_sha256": sha256_file(args.preprocessor),
        "preprocessor_output_dim": int(preprocessor.output_dim),
        "inversion_seconds": inversion_seconds,
        "total_seconds": total_seconds,
        "score_direction": "higher_is_more_anomalous",
        "completed_utc": utc_now(),
    }
    if split.labels is not None:
        result["metrics"] = compute_metrics(
            scores, split.labels, normal_label=args.normal_label)
    atomic_json(output_dir / "result.json", result)
    if "metrics" in result:
        print(
            "%s: samples=%d EER=%.4f%% AUC=%.6f" %
            (split.name, len(split.ids),
             100.0 * result["metrics"]["eer"],
             result["metrics"]["auc"]))
    else:
        print("%s: wrote %d scores" % (split.name, len(split.ids)))
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an AnoGAN checkpoint on any feature split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessor", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal-label", type=int, default=1)
    parser.add_argument(
        "--chunk-size", type=int, default=20000,
        help="raw rows transformed and scored at once")
    return parser.parse_args()


def main():
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
