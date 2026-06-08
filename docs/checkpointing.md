# Checkpointing

Training uses GoMLX's native context checkpoint format. The output path is fixed in code and is not a CLI flag.

```text
checkpoints/
  .gitkeep
  latest-run.txt
  runs/
    .gitkeep
    run-20260608-062617.123456789/
      manifest.json
      gomlx/
        checkpoint-n0000000-...json
        checkpoint-n0000000-...bin
```

By default, each command invocation starts a fresh run under `checkpoints/runs/<run-id>/`. Resume is opt-in: pass
`--resume` to load the run id recorded in `checkpoints/latest-run.txt` and continue from that run's latest GoMLX
checkpoint.

The run's `gomlx/` directory is passed to `github.com/gomlx/gomlx/pkg/ml/context/checkpoints`. GoMLX writes one JSON
metadata file and one binary tensor payload per checkpoint. The binary payload is compressed by GoMLX by default. The
trainer keeps the latest three GoMLX checkpoints within a run.

`manifest.json` is written by this project after each epoch inside the current run directory. It records the run id,
latest GoMLX checkpoint name, global step, split summaries, action vocabulary source, trainer configuration,
backend/device description, loss summaries, and VCS settings when available.

The trainer configuration includes the initial learning rate and whether GoMLX SGD learning-rate decay was enabled for
the run.

When a checkpoint is saved, `checkpoints/latest-run.txt` is updated to the current run id. A later command only resumes
that run if `--resume` is present. Without `--resume`, existing checkpoint files are ignored and a new run directory is
created.

The repository ignores generated checkpoint and telemetry contents. `checkpoints/.gitkeep` and
`checkpoints/runs/.gitkeep` are tracked so the output directories exist in a fresh checkout.
