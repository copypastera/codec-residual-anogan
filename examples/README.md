# Dataset adapter example

`manifest.csv` demonstrates the only dataset adapter the pipeline needs.
Replace its rows with your audio files. Paths may be absolute or relative to
the manifest.

Use any split names, but configure one or more training splits and one labeled
validation split in `configs/best.yaml`. A blind evaluation split can omit the
`label` values.

```bash
python scripts/extract.py --manifest examples/manifest.csv \
  --manifest-split train --output-dir data/features \
  --output-split train

python scripts/extract.py --manifest examples/manifest.csv \
  --manifest-split validation --output-dir data/features \
  --output-split validation
```

The extractor preserves manifest row order and refuses duplicate IDs, mixed
presence/absence of labels, or incompatible resume metadata.
