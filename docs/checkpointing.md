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
      tensorboard/
        events.out.tfevents....
```

By default, each command invocation starts a fresh run under `checkpoints/runs/<run-id>/`. Resume is opt-in: pass
`--resume` to load the run id recorded in `checkpoints/latest-run.txt` and continue from that run's latest GoMLX
checkpoint.

New runs can be given a short TensorBoard-visible label with `--run-label`. The
label is sanitized and appended to the timestamped run id:

```sh
go run ./cmd/train --run-label "small trunk 1x256 1x128"
```

```text
checkpoints/runs/run-20260609-181031.987654321-small-trunk-1x256-1x128/
```

Do not pass `--run-label` with `--resume`; resumed runs keep their existing run
id and TensorBoard key.

The run's `gomlx/` directory is passed to `github.com/gomlx/gomlx/pkg/ml/context/checkpoints`. GoMLX writes one JSON
metadata file and one binary tensor payload per checkpoint. The binary payload is compressed by GoMLX by default. The
trainer keeps the latest three GoMLX checkpoints within a run.

`manifest.json` is written by this project after each epoch inside the current run directory. It records the run id,
optional run label, latest GoMLX checkpoint name, global step, split summaries, action vocabulary source, trainer configuration,
backend/device description, TensorBoard event-file path, loss summaries, and VCS settings when available.

The trainer configuration includes the initial learning rate and whether GoMLX SGD learning-rate decay was enabled for
the run.

The run's `tensorboard/` directory contains scalar event files for live training progress. TensorBoard reads them through
the shared `checkpoints/runs/` directory used by `docker-compose.tensorboard.yml`.

When a checkpoint is saved, `checkpoints/latest-run.txt` is updated to the current run id. A later command only resumes
that run if `--resume` is present. Without `--resume`, existing checkpoint files are ignored and a new run directory is
created.

The repository ignores generated checkpoint and telemetry contents. `checkpoints/.gitkeep` and
`checkpoints/runs/.gitkeep` are tracked so the output directories exist in a fresh checkout.
