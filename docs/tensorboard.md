# TensorBoard

TensorBoard runs in a Docker container and reads event files from the shared host directory:

```text
checkpoints/runs/
```

The compose file binds the UI to localhost only:

```text
127.0.0.1:6006
```

Start it with:

```sh
make tensorboard-up
```

The Makefile creates `.env` from `.env.example` if `.env` is missing. The tracked defaults use UID/GID `1001`; set
`DOCKERCOMPOSE_UID` and `DOCKERCOMPOSE_GID` in the local, ignored `.env` file to match the host user that owns
`checkpoints/runs/`.

Then open:

```text
http://127.0.0.1:6006
```

Stop and remove the container with:

```sh
make tensorboard-trash
```

The TensorBoard container seeds a dummy run at startup if no dummy event file already exists:

```text
checkpoints/runs/dummy-backprop-run/
```

The dummy run writes TensorBoard event files, not JSON or SQLite. It includes two scalar series:

- `validation_delta_from_start`
- `learning_rate`

Real training runs write TensorBoard event files under:

```text
checkpoints/runs/<run-id>/tensorboard/
```

The training CLI prints the exact event-file path at startup:

```text
telemetry: tensorboard_dir="..." event_file="..."
```

TensorBoard watches `checkpoints/runs/`, so a run appears in the UI as soon as
training has created its event file. Because event files are under each run's
`tensorboard/` directory, TensorBoard shows real runs with names like:

```text
run-20260609-094514.633510746/tensorboard
```

The scalar series currently written by `cmd/train` are:

- `epoch`
- `learning_rate`
- `validation_delta_from_start`
- `train/mean_loss`
- `train/first_loss`
- `train/last_loss`
- `train/batches`
- `train/samples`
- `train/batches_per_second`
- `train/samples_per_second`
- `train/progress_loss`
- `train/progress_mean_loss`
- `train/progress_batches`
- `train/progress_samples`
- `train/progress_batches_per_second`
- `train/progress_samples_per_second`
- `validation/mean_loss`
- `validation/first_loss`
- `validation/last_loss`
- `validation/batches`
- `validation/samples`
- `validation/batches_per_second`
- `validation/samples_per_second`

The `train/progress_*` series are written at the same cadence as stdout progress logs, controlled by `--log-every`.
Set `--log-every 0` to disable both stdout progress logs and progress telemetry.

The dummy run is only a container smoke test and can be ignored when real run
data exists.
