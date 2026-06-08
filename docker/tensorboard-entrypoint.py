#!/usr/bin/env python3
import math
import os
import sys
import time

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def scalar_event(tag, step, value, wall_time):
    summary = summary_pb2.Summary(
        value=[
            summary_pb2.Summary.Value(
                tag=tag,
                simple_value=float(value),
            )
        ]
    )
    return event_pb2.Event(wall_time=wall_time, step=int(step), summary=summary)


def seed_dummy_backprop_run(run_dir):
    os.makedirs(run_dir, exist_ok=True)
    existing = [
        name
        for name in os.listdir(run_dir)
        if name.startswith("events.out.tfevents.")
    ]
    if existing:
        return

    writer = EventFileWriter(run_dir)
    started = time.time() - 600
    writer.add_event(event_pb2.Event(wall_time=started, file_version="brain.Event:2"))

    initial_lr = 0.01
    validation_deltas = [
        0.0,
        -0.42,
        -0.94,
        -1.31,
        -1.88,
        -2.14,
        -2.37,
        -2.28,
        -2.03,
        -1.76,
    ]
    for epoch, validation_delta in enumerate(validation_deltas):
        step = epoch * 1443
        wall_time = started + epoch * 60
        lr = initial_lr / math.sqrt(max(1, step)) if step else initial_lr
        writer.add_event(scalar_event("validation_delta_from_start", step, validation_delta, wall_time))
        writer.add_event(scalar_event("learning_rate", step, lr, wall_time))

    writer.flush()
    writer.close()


def main():
    logdir = os.environ.get("TENSORBOARD_LOGDIR", "/tensorboard/runs")
    host = os.environ.get("TENSORBOARD_HOST", "0.0.0.0")
    port = os.environ.get("TENSORBOARD_PORT", "6006")
    dummy_run_dir = os.environ.get("TENSORBOARD_DUMMY_RUN_DIR", os.path.join(logdir, "dummy-backprop-run"))

    seed_dummy_backprop_run(dummy_run_dir)

    args = [
        "tensorboard",
        "--logdir",
        logdir,
        "--host",
        host,
        "--port",
        port,
    ]
    os.execvp(args[0], args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"tensorboard entrypoint failed: {exc}", file=sys.stderr)
        raise
