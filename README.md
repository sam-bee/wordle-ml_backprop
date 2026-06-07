# Wordle Backprop Training

This project trains a Wordle neural network using GoMLX.

The goal is to use supervised backpropagation to make a model imitate pre-generated teacher data. The training data
contains incomplete Wordle game states and target outputs produced by a stronger teacher system.

## Initial Goals

1. Create a minimal Go project under `backprop`.
2. Add GoMLX as the machine learning library.
3. Load the synthetic training, validation, and test data.
4. Define a neural network matching the intended Wordle model shape closely enough for supervised learning.
5. Train the model with backpropagation.
6. Report training and validation loss during training.
7. Save enough model state that future work can inspect or reuse the trained model.

## Non-Goals For Now

This phase does not need to:

* integrate with CUDA directly;
* implement custom CUDA kernels;
* optimise inference speed;
* solve the full talk integration story.

Those are later phases.

## Expected Project Shape

The project should begin with a simple structure like:

```text
backprop/
  README.md
  go.mod
  cmd/
    train/
      main.go
  internal/
    data/
    model/
    training/
```

This structure can change if GoMLX examples suggest a better local convention, but the first implementation should stay
small and easy to understand.

## Training Data

The training data is expected to be generated before this project runs.

The intended data split is:

```text
data/
  train/
  validation/
  test/
```

Each split should contain structured records representing incomplete Wordle game states and the teacher output for that
state.

The exact file format may be binary with JSON metadata. The training code should avoid hard-coding assumptions that make
it difficult to update the data format later.

## Model

The model should be implemented in Go using GoMLX.

The starting point should be a straightforward supervised model, not a reinforcement learning system.

The model should accept an encoded incomplete Wordle game state and produce an output suitable for imitating the
teacher. The first version may use a simplified output target if that gets the training loop working sooner.

## Development Principle

Prefer a small working training loop over a large incomplete architecture.

The first milestone is:

> load a small batch, run a forward pass, compute a loss, and update the model parameters.

Once that works, the architecture, data loading, metrics, and persistence can be improved incrementally.
