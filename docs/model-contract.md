# Model/Data Contract

This document separates the data currently parsed by Go code from the intended model input contract in
`docs/todo/model-input-spec.md`. The parser and batcher preserve more fields than the real policy model should consume.

## Parsed Sample

The parser reads binary format version `2` records into `data.Sample`. Each record is 234 bytes and uses the constants
`MaxTurns = 5`, `WordLength = 5`, and `TopK = 16`.

| Field | Go type | Shape / size | Meaning |
| --- | --- | ---: | --- |
| `SolutionWord` | `data.Word` (`[5]byte`) | `word[5]` | Hidden solution word used to generate the record. It is loaded for auditing and split/evaluation checks. It is not model input. The train opening-state record uses the empty all-zero word. |
| `TurnDepth` | `uint8` | scalar | Number of previous turns in the visible state. The train opening state has `0`; non-opening records use `1..5`. The intended model does not consume this as a scalar turn-number feature, but it is needed to decide which previous-turn slots are occupied and to derive the virgin-grid flag. |
| `PreviousGuessWords` | `[5]data.Word` | `[5,5]` bytes | Five fixed-width previous-guess slots in chronological order. Only slots before `TurnDepth` are occupied turns; later slots are all-zero padding. |
| `PreviousFeedback` | `[5][5]data.Feedback` | `[5,5]` bytes | Feedback for each previous guess position. The file stores used turns as `0` grey, `1` yellow, or `2` green. Unused turns contain padding value `255`. The intended model encoder must remap used feedback values into the model-spec order green/yellow/grey. |
| `ShortlistSizeBefore` | `uint16` | scalar | Number of solution words still compatible with the visible state before the teacher selects the next guess. This is parsed and batched for inspection or possible loss weighting, but the model input spec excludes candidate constraints computed from the grid, so this should not be fed to the real policy model. |
| `TopKGuessWords` | `[16]data.Word` | `[16,5]` bytes | Teacher's ranked legal next guesses, best first. |
| `TopKReductionRatios` | `[16]float32` | `[16]` | Teacher score per ranked guess. The file-format doc defines this as `1.0 - worst_case_size / shortlist_size_before`; higher is better within a record. |
| `TopKWorstCaseSizes` | `[16]uint16` | `[16]` | Worst-case remaining shortlist size for each ranked teacher guess. |

No parser/file-format mismatch is currently known: the parser constants and loaded fields match
`docs/training-data-format.md`.

## Batch Contract

`data.BuildBatch` converts a slice of parsed samples into:

| Batch field | Go type | Batch shape | Meaning |
| --- | --- | ---: | --- |
| `Batch.Inputs[].TurnDepth` | `uint8` | `[B]` | Visible turn count for each sample. |
| `Batch.Inputs[].PreviousGuessWords` | `[5]data.Word` | `[B,5,5]` | Fixed previous-guess slots for each sample. |
| `Batch.Inputs[].PreviousFeedback` | `[5][5]data.Feedback` | `[B,5,5]` | Fixed feedback slots for each sample. |
| `Batch.Inputs[].ShortlistSizeBefore` | `uint16` | `[B]` | Compatible shortlist size before the teacher guess. Preserved in the batch, but excluded from the intended model input. |
| `Batch.Targets[].TopKGuessWords` | `[16]data.Word` | `[B,16,5]` | Ranked teacher guess words. |
| `Batch.Targets[].TopKReductionRatios` | `[16]float32` | `[B,16]` | Ranked teacher scores. |
| `Batch.Targets[].TopKWorstCaseSizes` | `[16]uint16` | `[B,16]` | Ranked teacher worst-case sizes. |

`B` is the actual batch size. The sequential batch iterator includes the final partial batch, so `B` may be smaller than
the configured batch size for the last batch.

## Intended Model Input

The intended GoMLX model should follow `docs/todo/model-input-spec.md`, which describes the input contract from the
similar CUDA implementation.

The model consumes a Wordle decision state before the next guess. It includes:

- up to five previous guess words;
- tile feedback for each previous guess;
- one scalar virgin-grid flag.

The intended model input excludes:

- `SolutionWord`;
- `ShortlistSizeBefore` and other candidate-word constraints computed from the grid;
- recurrent hidden state;
- attention masks;
- a separate scalar turn number, other than chronological slot order and the virgin-grid flag.

### Raw Turn Features

Each occupied previous-turn slot is encoded as a 145-value binary vector:

```text
5 letter positions * 26 letters = 130 values
5 feedback positions * 3 colours = 15 values
total = 145 values
```

Letter features use uppercase alphabet indices:

```text
A -> 0
B -> 1
...
Z -> 25
```

For a letter at position `p` and alphabet index `l`:

```text
letter_feature_index = (p * 26) + l
```

Feedback features occupy indices `130..144` using the model-spec order green, yellow, grey:

```text
green  -> 0
yellow -> 1
grey   -> 2
```

The parsed data stores feedback in a different numeric order. The required conversion is:

| Parsed value | Parsed meaning | Model feedback index |
| ---: | --- | ---: |
| `2` | green | `0` |
| `1` | yellow | `1` |
| `0` | grey | `2` |

For feedback at position `p` and model feedback index `f`:

```text
feedback_feature_index = 130 + (p * 3) + f
```

Each occupied turn should therefore set exactly 10 raw features to `1.0`: one letter feature and one feedback feature
for each of the five tile positions.

### Shared Turn Encoder

Each occupied turn's 145-value raw feature vector passes through the same shared turn encoder:

```text
145 -> 128 -> 64
```

Use ReLU after the 128-value hidden layer. Do not apply an activation to the 64-value encoder output. The same encoder
weights are reused for every occupied turn slot.

Empty slots must not be encoded by passing an all-zero 145-value vector through the shared encoder. For every empty
slot, the model input uses a literal 64-value zero vector.

### Dense Trunk Input

After turn encoding, each sample has a 321-value dense-trunk input:

```text
1 virgin-grid flag + (5 turn slots * 64 encoded values) = 321 values
```

For a batch, the trunk input shape is:

```text
[B,321]
```

Index `0` is the virgin-grid flag:

```text
input[0] = 1.0 if TurnDepth == 0
input[0] = 0.0 otherwise
```

The five encoded turn slots follow in chronological order:

```text
slot 0 starts at input[1]
slot 1 starts at input[65]
slot 2 starts at input[129]
slot 3 starts at input[193]
slot 4 starts at input[257]
```

The dense trunk described by the spec is:

```text
321 -> 256 -> 128 -> 64
```

Use ReLU after the 256-value layer and after the 128-value layer. Do not apply an activation to the final 64-value
vector.

## Current GoMLX Sanity Check

The current GoMLX code is only a temporary training sanity check. It proves that parsed batches can become GoMLX tensors
and that one update step can run. It is not the intended model input contract.

`training.BatchToTensors` currently converts one `data.Batch` into:

| Tensor | DType | Shape | Source |
| --- | --- | ---: | --- |
| input | `float32` | `[B,52]` | Temporary flattened numeric features. |
| label | `float32` | `[B,16]` | `TopKReductionRatios` only. |

The current temporary input feature order is:

1. `TurnDepth / 5`
2. `ShortlistSizeBefore / 2309`
3. all `PreviousGuessWords` bytes in slot/position order, encoded as `0` for padding and `A..Z` as `1..26 / 26`
4. all `PreviousFeedback` values in turn/position order, encoded as green `1`, yellow `0.5`, and everything else `0`

`model.SanityModel` expects one input tensor with at least rank 2, reshapes it to `[B, -1]`, requires exactly `52`
features, and applies one dense layer with bias to produce `[B,16]`.

`training.RunSanityStep` uses:

| Component | Current choice |
| --- | --- |
| backend | GoMLX SimpleGo backend |
| model | one dense linear layer |
| target | teacher reduction ratios only |
| loss | GoMLX mean squared error |
| optimizer | SGD, learning rate `0.05`, no decay |
| execution | initial eval loss, one train step, post-update eval loss |

## Known Gaps Before The Real Model

- Replace the `[B,52]` sanity tensor with the spec-compatible occupied-turn encoder and `[B,321]` dense-trunk input.
- Remove `TurnDepth` as a direct numeric model input. Use it only to identify occupied slots and derive the virgin-grid
  flag.
- Remove `ShortlistSizeBefore` from real model inputs. The training-data docs say it is optionally usable, but the
  model input spec explicitly excludes candidate constraints computed from the grid.
- Replace the temporary ASCII-normalized word-byte encoding with per-position `A..Z` one-hot features.
- Replace the temporary feedback scalar encoding with the spec one-hot order green/yellow/grey, including the required
  remap from parsed file values.
- Ensure empty slots bypass the shared turn encoder and contribute literal 64-value zero vectors.
- Keep `SolutionWord` excluded from model inputs.
- Decide how the parsed top-16 teacher labels should supervise the final policy output. The input spec defines the
  model input and trunk shape, while the current sanity check trains only `[B,16]` reduction-ratio regression.
