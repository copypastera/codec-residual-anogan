# Codec-Residual AnoGAN

Dataset-agnostic, end-to-end codec-residual AnoGAN for replay and audio
anomaly detection. This repository provides one reproducible pipeline:

```text
audio manifest -> codec-residual extraction -> preprocessing -> AnoGAN training
               -> validation checkpoint selection -> evaluation scores/EER/AUC
```

The repository intentionally contains only the best AnoGAN recipe. It does
not include OCSVM baselines, abandoned experiment branches, benchmark-specific
executables, or test files.

## Released model

| Split role | EER |
|---|---:|
| Model-selection split | **21.0900%** |
| Independent evaluation split | **22.6442%** |

The released model is epoch 4. It uses StandardScaler, 98%-variance PCA
(512 to 287 dimensions), a convolutional AnoGAN, and discriminator spectral
normalization. Larger scores mean more likely anomalous/replay audio.

## Repository layout

```text
artifacts/best/
  checkpoint.pt                 released epoch-4 model
  preprocessor.joblib           released scaler and PCA
  preprocessor.json             preprocessing summary
  metadata.json                 hashes, architecture, and reported results
configs/best.yaml               reusable best training configuration
examples/manifest.csv           minimal dataset manifest
scripts/
  extract.py                    source-tree extraction entry point
  train.py                      source-tree training entry point
  evaluate.py                   source-tree evaluation entry point
src/codec_residual_anogan/
  features.py                   WORLD/Opus codec-residual extraction
  data.py                       generic split loading and validation
  preprocessing.py              scaling and PCA
  models.py                     generator and discriminator
  training.py                   BCE AnoGAN training
  inference.py                  latent inversion and anomaly scoring
  metrics.py                    EER and AUC
  cli/                          installed command-line entry points
environment.yml                reference Conda environment
pyproject.toml                  Python package and console commands
```

## 1. Clone the repository

Replace `<REPOSITORY_URL>` with the URL where this folder is published:

```bash
git clone <REPOSITORY_URL> codec-residual-anogan
cd codec-residual-anogan
```

Run all commands below from the repository root. Relative paths in the YAML
configuration are resolved from the current working directory.

## 2. Install the environment

### Option A: reference Conda environment

The reference environment uses Python 3.8, PyTorch 1.10.2, CUDA 11.3, and
scikit-learn 0.24.2:

```bash
conda env create -f environment.yml
conda activate codec-residual-anogan
python -m pip install -e .
```

### Option B: an existing Python environment

Python 3.8 or newer is supported. Create or activate an environment, install
a PyTorch build suitable for your CPU/CUDA system, then install this package:

```bash
python -m pip install -e ".[extract]"
```

Feature extraction also requires an `ffmpeg` executable with `libopus`:

```bash
ffmpeg -hide_banner -encoders | grep libopus
```

Confirm that the three installed commands are available:

```bash
cra-extract --help
cra-train --help
cra-evaluate --help
```

If you do not install the package, use the equivalent source-tree commands:

```text
cra-extract   -> python scripts/extract.py
cra-train     -> python scripts/train.py
cra-evaluate  -> python scripts/evaluate.py
```

## 3. Prepare a dataset manifest

The pipeline does not assume a particular dataset directory or protocol.
Describe every audio file in a CSV or TSV manifest with these columns:

| Column | Required | Meaning |
|---|---|---|
| `id` | yes | Unique sample identifier |
| `path` | yes | Audio path |
| `label` | optional | `1` = normal/bona fide, `0` = anomaly/replay |
| `split` | optional | Split selected by `--manifest-split` |

For example, place local audio under the ignored `data/` directory:

```bash
mkdir -p manifests data/my_dataset/audio
```

Then arrange the files like this:

```text
data/my_dataset/audio/train_normal_0001.wav
data/my_dataset/audio/train_normal_0002.wav
data/my_dataset/audio/validation_normal_0001.wav
data/my_dataset/audio/validation_replay_0001.wav
data/my_dataset/audio/evaluation_0001.wav
manifests/my_dataset.csv
```

Create `manifests/my_dataset.csv`:

```csv
id,path,label,split
train_normal_0001,../data/my_dataset/audio/train_normal_0001.wav,1,train
train_normal_0002,../data/my_dataset/audio/train_normal_0002.wav,1,train
validation_normal_0001,../data/my_dataset/audio/validation_normal_0001.wav,1,validation
validation_replay_0001,../data/my_dataset/audio/validation_replay_0001.wav,0,validation
evaluation_0001,../data/my_dataset/audio/evaluation_0001.wav,,evaluation
```

Relative audio paths are resolved from the manifest directory, not from the
repository root. Absolute paths are also accepted. WAV, FLAC, and other
libsndfile-supported formats can be used. Stereo audio is mixed to mono and
audio at other sample rates is resampled to 16 kHz.

Manifest rules:

- IDs must be non-empty and unique within each selected split.
- Labels must be present for every row in a selected split or absent for all
  rows in that split.
- A labeled validation split must contain both `0` and `1`.
- A labeled training split contributes only rows matching `normal_label`.
- An unlabeled training split is treated as entirely normal.
- Evaluation may be labeled or unlabeled.

## 4. Extract codec-residual features

Create a feature directory and extract each split independently:

```bash
mkdir -p data/my_dataset/features

cra-extract \
  --manifest manifests/my_dataset.csv \
  --manifest-split train \
  --output-dir data/my_dataset/features \
  --output-split train \
  --bitrate 16k \
  --workers 8

cra-extract \
  --manifest manifests/my_dataset.csv \
  --manifest-split validation \
  --output-dir data/my_dataset/features \
  --output-split validation \
  --bitrate 16k \
  --workers 8

cra-extract \
  --manifest manifests/my_dataset.csv \
  --manifest-split evaluation \
  --output-dir data/my_dataset/features \
  --output-split evaluation \
  --bitrate 16k \
  --workers 8
```

Use a worker count appropriate for the machine. Extraction is indexed,
order-preserving, and resumable. If it is interrupted, run the exact same
command again.

For each output split, the important files are:

```text
<split>_residual.npy   N x 512 float32 feature matrix
<split>_ids.txt        one sample ID for each feature row
<split>_labels.txt     ID and label when the manifest split is labeled
<split>_done.npy       extraction resume state
<split>_status.json    progress summary
<split>_errors.json    per-sample extraction errors
<split>_manifest.json  extraction parameters and provenance
```

Extraction must finish without unresolved errors before training or
evaluation. Reuse the same output split only when resuming the same manifest;
use a new output name or directory after changing IDs or labels.

### Feature definition

Each utterance becomes one 512-D vector:

1. Analyze and resynthesize the waveform with WORLD
   (Harvest, CheapTrick, and D4C).
2. Encode/decode the result with Opus at 16 kb/s in VoIP mode.
3. Compute a 1024-point STFT with an 800-sample window and 400-sample hop.
4. Temporally average the first 512 log-magnitude bins.
5. Store `original spectrum - processed spectrum`.

## 5. Configure a training run

Copy the best recipe instead of editing the shared template:

```bash
cp configs/best.yaml configs/my_dataset.yaml
```

Edit these fields in `configs/my_dataset.yaml`:

```yaml
run:
  name: my_dataset_seed42
  output_dir: runs/my_dataset_seed42
  seed: 42

data:
  feature_directory: data/my_dataset/features
  train_splits:
    - train
  validation_split: validation
  normal_label: 1

runtime:
  device: cuda:0       # use cpu when CUDA is unavailable
  memory_fraction: 0.85
  deterministic_inference: false
```

`train_splits` can contain multiple extracted split names. Only normal rows
are used to fit the preprocessor and train AnoGAN. Keep the preprocessing,
model, training, inference, and anomaly-score sections unchanged to use the
released best recipe.

The configured `run.output_dir` must not already exist. This prevents an old
run from being silently overwritten. Choose a new directory for every run.

## 6. Train and select the best epoch

Create a separate log directory and start training:

```bash
mkdir -p logs
cra-train --config configs/my_dataset.yaml 2>&1 | tee logs/my_dataset_seed42.log
```

Do not place the terminal log inside `run.output_dir` before training, because
the trainer requires that output directory to be absent.

During validation, console output looks like:

```text
validation batch=1/170 seen=512 elapsed=4.2s
validation batch=20/170 seen=10240 elapsed=78.5s
epoch=001 g_loss=0.712345 d_loss=1.284012 validation_eer=24.3521% threshold=0.81234567
```

The trainer evaluates the labeled validation split after every epoch and
selects the lowest validation EER. Generated run contents:

```text
runs/my_dataset_seed42/
  config.yaml
  metrics.csv
  selection.json
  best_validation_scores.npz
  best_checkpoint.pt
  checkpoints/
    epoch_001.pt
    ...                         retained top-k epoch checkpoints
  preprocessing/
    preprocessor.joblib
    preprocessor.json
```

`metrics.csv` is the structured epoch log. `selection.json` records the best
epoch, validation EER, threshold, and path to `best_checkpoint.pt`. Early
stopping occurs after the configured number of non-improving epochs.

## 7. Evaluate a trained model

Score the extracted evaluation split with the checkpoint and preprocessor
from the same training run:

```bash
cra-evaluate \
  --checkpoint runs/my_dataset_seed42/best_checkpoint.pt \
  --preprocessor runs/my_dataset_seed42/preprocessing/preprocessor.joblib \
  --feature-dir data/my_dataset/features \
  --split evaluation \
  --output-dir results/my_dataset_seed42_evaluation \
  --device cuda:0 \
  --memory-fraction 0.55 \
  --chunk-size 20000 \
  --seed 42 \
  --normal-label 1
```

For CPU evaluation, replace `--device cuda:0` with `--device cpu` and set
`--memory-fraction 0`. Reduce `--chunk-size` if host RAM is limited.

The evaluation output directory must not already exist. It contains:

```text
results/my_dataset_seed42_evaluation/
  scores.npz
  result.json
```

`scores.npz` preserves manifest order and includes sample IDs, anomaly scores,
latent-inversion components, and labels when available. If labels exist,
`result.json` includes EER, EER threshold, AUC, score statistics, artifact
hashes, and elapsed time. For an unlabeled split, scoring still succeeds and
writes the ordered scores without metrics.

## 8. Evaluate the released checkpoint

The included artifact can be scored with the same generic evaluator:

```bash
cra-evaluate \
  --checkpoint artifacts/best/checkpoint.pt \
  --preprocessor artifacts/best/preprocessor.joblib \
  --feature-dir data/my_dataset/features \
  --split evaluation \
  --output-dir results/released_checkpoint_evaluation \
  --device cuda:0 \
  --chunk-size 20000 \
  --seed 42
```

The released preprocessor was fitted on the original training data. For a fair
new-dataset experiment, train a new model and use its matching preprocessor.
The released checkpoint is primarily included for result provenance and
reference scoring.

## Reproducing the reported result

`artifacts/best/metadata.json` records the selected epoch, architecture,
preprocessing, anomaly-score formula, thresholds, reported EER/AUC, and exact
SHA-256 hashes. The original licensed benchmark audio is not distributed.
Reproducing the reported numbers requires the same source audio, split
assignment, and row ordering, prepared through the interface above.

## Common problems

- **`CUDA was requested but is unavailable`:** set `runtime.device: cpu` in
  the training YAML or pass `--device cpu` during evaluation.
- **`output directory exists`:** choose a new `run.output_dir` or evaluation
  `--output-dir`. Existing results are never overwritten.
- **Missing `libopus`:** install an FFmpeg build that includes the encoder and
  verify it with `ffmpeg -encoders`.
- **Empty manifest selection:** make sure `--manifest-split` exactly matches
  the value in the manifest `split` column.
- **Existing split metadata differs:** use a new output directory or split
  name after changing the manifest.
- **Validation label error:** validation must contain both normal (`1`) and
  anomaly (`0`) samples.
- **Out of GPU memory:** reduce inference `batch_size` in the YAML or lower
  evaluation `--chunk-size`.

No sample counts or dataset names are hard-coded in the package. Generated
features, runs, results, and logs are ignored by Git.
