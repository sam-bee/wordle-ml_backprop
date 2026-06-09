# Changes to Model Architecture

This document exhaustively lists any changes to the model architecture, since the original implementation. Its purpose
is to inform model architecture decisions in a parallel project. It may be necessary for any changes to the architecture
to be implemented in another programming language, so this document aims to specify any alterations to the model
architecture in sufficient detail to support such changes.

## 2026-06-08: Dense Trunk Depth Experiment

The first dense-trunk depth experiment added two extra 128-neuron hidden layers between the original 128-neuron hidden
layer and the 64-value policy output head.

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

New GoMLX variable scopes for that experiment:

- `dense_trunk.hidden1_to_hidden2`
- `dense_trunk.hidden2_to_hidden3`
- `dense_trunk.hidden3_to_output`

The old final trunk scope `dense_trunk.hidden1_to_output` is replaced by `dense_trunk.hidden3_to_output` in this
experiment.

Dense-trunk parameter count changes from 123,584 trainable scalars to 156,608. The policy-model dense parameter
count changes from 150,528 scalars to 183,552.

## 2026-06-09: Additional 256-Wide Dense Trunk Layer

The dense-trunk depth experiment was extended by inserting one 256-neuron hidden layer between the existing 256-neuron
layer and the first 128-neuron layer.

Previous experimental dense trunk:

```text
321 -> 256 -> 128 -> 128 -> 128 -> 64
```

Updated experimental dense trunk:

```text
321 -> 256 -> 256 -> 128 -> 128 -> 128 -> 64
```

The inserted 256-neuron layer uses ReLU activation. The final 64-value policy vector remains linear, with no output
activation or normalization.

Updated GoMLX variable scopes:

- `dense_trunk.hidden0_to_hidden1` is now `256 -> 256`
- `dense_trunk.hidden1_to_hidden2` is now `256 -> 128`
- `dense_trunk.hidden2_to_hidden3` remains `128 -> 128`
- `dense_trunk.hidden3_to_hidden4` is a new `128 -> 128` layer
- `dense_trunk.hidden4_to_output` replaces `dense_trunk.hidden3_to_output` as the final projection

Dense-trunk parameter count changes from 156,608 trainable scalars to 222,400. The policy-model dense parameter
count changes from 183,552 scalars to 249,344.

## 2026-06-09: Two Additional 256-Wide Dense Trunk Layers

The dense-trunk depth experiment was extended again by inserting two more 256-neuron hidden layers alongside the
existing 256-neuron layers, before the first 128-neuron layer.

Previous experimental dense trunk:

```text
321 -> 256 -> 256 -> 128 -> 128 -> 128 -> 64
```

Updated experimental dense trunk:

```text
321 -> 256 -> 256 -> 256 -> 256 -> 128 -> 128 -> 128 -> 64
```

The inserted 256-neuron layers use ReLU activation. The final 64-value policy vector remains linear, with no output
activation or normalization.

Updated GoMLX variable scopes:

- `dense_trunk.hidden0_to_hidden1` remains `256 -> 256`
- `dense_trunk.hidden1_to_hidden2` is now `256 -> 256`
- `dense_trunk.hidden2_to_hidden3` is now `256 -> 256`
- `dense_trunk.hidden3_to_hidden4` is now `256 -> 128`
- `dense_trunk.hidden4_to_hidden5` is a shifted `128 -> 128` layer
- `dense_trunk.hidden5_to_hidden6` is a shifted `128 -> 128` layer
- `dense_trunk.hidden6_to_output` replaces `dense_trunk.hidden4_to_output` as the final projection

Dense-trunk parameter count changes from 222,400 trainable scalars to 353,984. The policy-model dense parameter count
changes from 249,344 scalars to 380,928.

## 2026-06-09: Two Additional 128-Wide Dense Trunk Layers

The dense-trunk depth experiment was extended again by inserting two more 128-neuron hidden layers alongside the
existing 128-neuron layers, before the 64-value policy output head.

Previous experimental dense trunk:

```text
321 -> 256 -> 256 -> 256 -> 256 -> 128 -> 128 -> 128 -> 64
```

Updated experimental dense trunk:

```text
321 -> 256 -> 256 -> 256 -> 256 -> 128 -> 128 -> 128 -> 128 -> 128 -> 64
```

The inserted 128-neuron layers use ReLU activation. The final 64-value policy vector remains linear, with no output
activation or normalization.

Updated GoMLX variable scopes:

- `dense_trunk.hidden4_to_hidden5` remains `128 -> 128`
- `dense_trunk.hidden5_to_hidden6` remains `128 -> 128`
- `dense_trunk.hidden6_to_hidden7` is a new `128 -> 128` layer
- `dense_trunk.hidden7_to_hidden8` is a new `128 -> 128` layer
- `dense_trunk.hidden8_to_output` replaces `dense_trunk.hidden6_to_output` as the final projection

Dense-trunk parameter count changes from 353,984 trainable scalars to 387,008. The policy-model dense parameter count
changes from 380,928 scalars to 413,952.
