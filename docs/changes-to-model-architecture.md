# Changes to Model Architecture

This document exhaustively lists any changes to the model architecture, since the original implementation. Its purpose
is to inform model architecture decisions in a parallel project. It may be necessary for any changes to the architecture
to be implemented in another programming language, so this document aims to specify any alterations to the model
architecture in sufficient detail to support such changes.

## 2026-06-08: Dense Trunk Depth Experiment

The GoMLX model has an experimental dense-trunk variant with two extra 128-neuron hidden layers between the original
128-neuron hidden layer and the 64-value policy output head.

Previous dense trunk:

```text
321 -> 256 -> 128 -> 64
```

Experimental dense trunk:

```text
321 -> 256 -> 128 -> 128 -> 128 -> 64
```

The two new layers both use ReLU activation. The final 64-value policy vector remains linear, with no output activation
or normalization.

New GoMLX variable scopes:

- `dense_trunk.hidden1_to_hidden2`
- `dense_trunk.hidden2_to_hidden3`
- `dense_trunk.hidden3_to_output`

The old final trunk scope `dense_trunk.hidden1_to_output` is replaced by `dense_trunk.hidden3_to_output` in this
experiment.

Dense-trunk parameter count changes from 123,584 fp16-equivalent scalars to 156,608. The policy-model dense parameter
count changes from 150,528 scalars to 183,552.
