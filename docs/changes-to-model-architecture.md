# Changes to Model Architecture

This document lists deliberate changes to the Wordle policy model architecture in this project.

It tracks structural changes such as input encoders, dense trunk layers, output embeddings, and policy heads. It does not
track training-loop, optimizer, checkpointing, telemetry, or CLI changes.

No model-architecture changes are recorded on `master` beyond the current baseline described in
`docs/model-architecture.md`.

## 2026-06-09: Input Encoder And 256-Wide Trunk Experiment

This branch adds one 64-neuron hidden layer to the shared per-turn input encoder and one 256-neuron hidden layer to the
dense trunk.

Previous shared input encoder:

```text
145 -> 128 -> 64
```

Updated shared input encoder:

```text
145 -> 128 -> 64 -> 64
```

The inserted encoder layer is `input_encoder.hidden_to_hidden64`, with shape `128 -> 64` and ReLU activation. The final
encoder projection is now `input_encoder.hidden64_to_output`, with shape `64 -> 64` and no activation.

Previous dense trunk:

```text
321 -> 256 -> 128 -> 64
```

Updated dense trunk:

```text
321 -> 256 -> 256 -> 128 -> 64
```

The inserted trunk layer is `dense_trunk.hidden0_to_hidden1`, with shape `256 -> 256` and ReLU activation. The previous
`256 -> 128` trunk layer is now `dense_trunk.hidden1_to_hidden2`, and the final projection is now
`dense_trunk.hidden2_to_output`.

Input-encoder parameter count changes from 26,944 trainable scalars to 31,104. Dense-trunk parameter count changes from
123,584 trainable scalars to 189,376. Dense policy network parameter count changes from 150,528 trainable scalars to
220,480.
