# Changes to Model Architecture

This document lists deliberate changes to the Wordle policy model architecture in this project.

It tracks structural changes such as input encoders, dense trunk layers, output embeddings, and policy heads. It does not
track training-loop, optimizer, checkpointing, telemetry, or CLI changes.

## smaller-embeddings

This experiment keeps the dense trunk's current 64-value output layer, adds one
extra 128-value hidden layer next to the existing 128-value layer, then adds a
separate `64 -> 48` policy output head.

The dense trunk is:

```text
321 -> 256 -> 128 -> 128 -> 64
```

The output embedding remains split into fixed letter-count features plus a
trainable tail:

```text
48 = 26 fixed word features + 22 trainable tail features
```

The 26 fixed letter-count features are unchanged. The trainable per-action tail
is reduced from 38 values to 22 values.
