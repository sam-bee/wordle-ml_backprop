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

Future training integration should write real scalar events under each run directory, for example:

```text
checkpoints/runs/<run-id>/tensorboard/
```
