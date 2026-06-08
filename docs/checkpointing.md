# Checkpointing

Training uses GoMLX's native context checkpoint format. The output path is fixed in code and is not a CLI flag.

```text
checkpoints/
  .gitkeep
  manifest.json
  gomlx/
    checkpoint-n0000000-...json
    checkpoint-n0000000-...bin
```

`checkpoints/gomlx/` is passed to `github.com/gomlx/gomlx/pkg/ml/context/checkpoints`. GoMLX writes one JSON metadata
file and one binary tensor payload per checkpoint. The binary payload is compressed by GoMLX by default. The trainer
keeps the latest three GoMLX checkpoints.

`checkpoints/manifest.json` is written by this project after each epoch. It records the latest GoMLX checkpoint name,
global step, split summaries, action vocabulary source, trainer configuration, backend/device description, loss
summaries, and VCS settings when available.

If checkpoint files already exist in `checkpoints/gomlx/`, GoMLX loads the latest checkpoint when the trainer starts.
The next completed epoch saves a new checkpoint and refreshes the manifest.

The repository ignores generated checkpoint contents. Only `checkpoints/.gitkeep` is tracked so the output directory
exists in a fresh checkout.
