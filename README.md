# Wordle Backprop Training

This project trains a Wordle neural network using GoMLX.

The goal is to use supervised backpropagation to make a model imitate pre-generated teacher data. The training data
contains incomplete Wordle game states and target outputs produced by a stronger teacher system.

## Current Status

The project currently:

1. Loads the synthetic training, validation, and test data.
2. Defines the intended Wordle policy model in GoMLX.
3. Trains the model with supervised backpropagation.
4. Reports training and validation loss during training.
5. Saves native GoMLX checkpoints plus a project manifest.
6. Provides a small play CLI for inspecting checkpoint-backed policy choices.
7. Provides a Dockerized TensorBoard container with a dummy telemetry run; real training-to-TensorBoard event writing is
   still a follow-up.

## Non-Goals For Now

This phase does not need to:

* integrate with CUDA directly;
* implement custom CUDA kernels;
* optimise inference speed;
* solve the full talk integration story.

Those are later phases.

## Project Shape

The current high-level structure is:

```text
backprop/
  README.md
  go.mod
  cmd/
    play/
    train/
  docker/
  docs/
  internal/
    data/
    model/
    training/
```

## Training Data

The training data is expected to be generated before this project runs.

The intended data split is:

```text
data/
  train/
  validation/
  test/
```

Each split contains fixed-width binary records plus JSON metadata. See `docs/training-data-format.md`.

## Model

The model is implemented in Go using GoMLX.

The training setup is a straightforward supervised model, not a reinforcement learning system.

The model accepts an encoded incomplete Wordle game state and produces action logits for imitating the teacher's ranked
next-guess choices. See `docs/model-contract.md`.

## Development Principle

Prefer a small working training loop over a large incomplete architecture.
