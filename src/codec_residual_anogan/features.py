"""Manifest-driven WORLD/Opus codec-residual feature extraction."""
import csv
import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .utils import atomic_json


SAMPLE_RATE = 16000
N_BINS = 512
_WORKER = {}


@dataclass
class ManifestRow:
    sample_id: str
    path: str
    label: object = None
    split: object = None


def read_manifest(path, split=None):
    """Read CSV/TSV columns: id, path, optional label, optional split."""
    path = Path(path)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {"id", "path"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("manifest requires id and path columns")
        for raw in reader:
            row_split = (raw.get("split") or "").strip() or None
            if split is not None and row_split != split:
                continue
            audio_path = Path(raw["path"].strip())
            if not audio_path.is_absolute():
                audio_path = path.parent / audio_path
            label_text = (raw.get("label") or "").strip()
            rows.append(ManifestRow(
                sample_id=raw["id"].strip(),
                path=str(audio_path.resolve()),
                label=int(label_text) if label_text else None,
                split=row_split))
    if not rows:
        raise RuntimeError("manifest selection is empty")
    ids = [row.sample_id for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("manifest has empty or duplicate IDs")
    labels_present = [row.label is not None for row in rows]
    if any(labels_present) and not all(labels_present):
        raise RuntimeError("manifest labels must be present for every row or none")
    return rows


def _audio_modules():
    try:
        import pyworld
        import soundfile
        from scipy.signal import resample_poly, stft
    except ImportError as error:
        raise RuntimeError(
            "feature extraction dependencies are missing; install "
            "codec-residual-anogan[extract]") from error
    return pyworld, soundfile, resample_poly, stft


def _average_log_spectrum(waveform):
    _, _, _, stft = _audio_modules()
    _, _, spectrum = stft(
        waveform, fs=SAMPLE_RATE, nperseg=800, noverlap=400, nfft=1024,
        boundary=None)
    return np.log(np.abs(spectrum[:N_BINS]) + 1e-8).mean(axis=1)


def _world_resynthesis(waveform):
    pyworld, _, _, _ = _audio_modules()
    values = np.ascontiguousarray(waveform, dtype=np.float64)
    f0, times = pyworld.harvest(values, SAMPLE_RATE)
    spectrum = pyworld.cheaptrick(values, f0, times, SAMPLE_RATE)
    aperiodicity = pyworld.d4c(values, f0, times, SAMPLE_RATE)
    return pyworld.synthesize(
        f0, spectrum, aperiodicity, SAMPLE_RATE)


def _read_audio(path):
    _, soundfile, resample_poly, _ = _audio_modules()
    waveform, sample_rate = soundfile.read(path)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = np.asarray(waveform, dtype=np.float64)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        waveform = resample_poly(
            waveform, SAMPLE_RATE // divisor, int(sample_rate) // divisor)
    if len(waveform) < 800 or not np.isfinite(waveform).all():
        raise ValueError("audio is too short or contains non-finite samples")
    return waveform


def _init_worker(bitrate):
    _audio_modules()
    base = "/dev/shm" if os.access("/dev/shm", os.W_OK) else None
    _WORKER["bitrate"] = bitrate
    _WORKER["temporary"] = tempfile.mkdtemp(
        prefix="codec_residual_", dir=base)


def _extract_one(item):
    index, sample_id, path = item
    try:
        _, soundfile, _, _ = _audio_modules()
        waveform = _read_audio(path)
        temporary = _WORKER["temporary"]
        world_path = os.path.join(temporary, "world.wav")
        opus_path = os.path.join(temporary, "encoded.opus")
        decoded_path = os.path.join(temporary, "decoded.wav")
        soundfile.write(
            world_path, _world_resynthesis(waveform), SAMPLE_RATE)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", world_path,
            "-c:a", "libopus", "-b:a", _WORKER["bitrate"],
            "-application", "voip", opus_path], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", opus_path,
            "-ar", str(SAMPLE_RATE), decoded_path], check=True)
        processed, processed_rate = soundfile.read(decoded_path)
        if processed_rate != SAMPLE_RATE:
            raise ValueError("decoded audio is not 16 kHz")
        if processed.ndim == 2:
            processed = processed.mean(axis=1)
        length = min(len(waveform), len(processed))
        residual = (
            _average_log_spectrum(waveform[:length])
            - _average_log_spectrum(processed[:length]))
        residual = np.asarray(residual, dtype=np.float32)
        if residual.shape != (N_BINS,) or not np.isfinite(residual).all():
            raise ValueError("invalid residual")
        return index, sample_id, residual, None
    except Exception as error:
        return index, sample_id, None, repr(error)


def _write_or_validate(path, content):
    path = Path(path)
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError("existing split metadata differs: %s" % path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    os.replace(str(temporary), str(path))


def _open_arrays(feature_path, done_path, count):
    if feature_path.exists() or done_path.exists():
        if not feature_path.exists() or not done_path.exists():
            raise RuntimeError("incomplete extraction cache")
        features = np.load(feature_path, mmap_mode="r+")
        done = np.load(done_path, mmap_mode="r+")
        if features.shape != (count, N_BINS) or done.shape != (count,):
            raise RuntimeError("existing extraction cache shape mismatch")
        return features, done
    temporary_features = Path(str(feature_path) + ".creating")
    temporary_done = Path(str(done_path) + ".creating")
    features = np.lib.format.open_memmap(
        temporary_features, mode="w+", dtype=np.float32,
        shape=(count, N_BINS))
    done = np.lib.format.open_memmap(
        temporary_done, mode="w+", dtype=np.bool_, shape=(count,))
    done[:] = False
    features.flush()
    done.flush()
    del features, done
    os.replace(str(temporary_features), str(feature_path))
    os.replace(str(temporary_done), str(done_path))
    return (
        np.load(feature_path, mmap_mode="r+"),
        np.load(done_path, mmap_mode="r+"))


def extract_manifest(manifest, output_dir, output_split, manifest_split=None,
                     bitrate="16k", workers=8, checkpoint_every=500):
    """Extract one deterministic feature split, resuming if interrupted."""
    rows = read_manifest(manifest, split=manifest_split)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / output_split
    ids_path = Path(str(prefix) + "_ids.txt")
    labels_path = Path(str(prefix) + "_labels.txt")
    feature_path = Path(str(prefix) + "_residual.npy")
    done_path = Path(str(prefix) + "_done.npy")
    status_path = Path(str(prefix) + "_status.json")
    errors_path = Path(str(prefix) + "_errors.json")
    manifest_path = Path(str(prefix) + "_manifest.json")

    _write_or_validate(
        ids_path, "".join("%s\n" % row.sample_id for row in rows))
    if rows[0].label is not None:
        _write_or_validate(
            labels_path,
            "".join(
                "%s %d\n" % (row.sample_id, row.label) for row in rows))
    features, done = _open_arrays(feature_path, done_path, len(rows))
    todo = [
        (index, row.sample_id, row.path)
        for index, row in enumerate(rows) if not bool(done[index])]
    atomic_json(manifest_path, {
        "source_manifest": str(Path(manifest).resolve()),
        "manifest_split": manifest_split,
        "output_split": output_split,
        "rows": len(rows),
        "sample_rate": SAMPLE_RATE,
        "feature_dimension": N_BINS,
        "bitrate": bitrate,
        "feature": "WORLD_Harvest_Opus16_temporal_average_residual",
    })
    if not todo:
        return feature_path

    started = time.time()
    errors = {}
    pending = []
    with Pool(
            workers, initializer=_init_worker, initargs=(bitrate,)) as pool:
        for processed, result in enumerate(
                pool.imap_unordered(_extract_one, todo, chunksize=2), 1):
            index, sample_id, residual, error = result
            if error is None:
                features[index] = residual
                pending.append(index)
            else:
                errors[sample_id] = error
            if processed % checkpoint_every == 0 or processed == len(todo):
                features.flush()
                done[np.asarray(pending, dtype=np.int64)] = True
                done.flush()
                pending.clear()
                elapsed = time.time() - started
                atomic_json(status_path, {
                    "total": len(rows),
                    "done": int(done.sum()),
                    "processed_this_run": processed,
                    "errors_this_run": len(errors),
                    "elapsed_seconds": elapsed,
                    "complete": bool(done.all()),
                })
                atomic_json(errors_path, errors)
                print(
                    "%d/%d processed; total_done=%d/%d; errors=%d" %
                    (processed, len(todo), int(done.sum()), len(rows),
                     len(errors)), flush=True)
    if not bool(done.all()):
        raise RuntimeError(
            "extraction incomplete; rerun the same command to retry failures")
    return feature_path
