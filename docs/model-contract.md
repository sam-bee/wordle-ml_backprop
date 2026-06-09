# Model/Data Contract

This document separates the data parsed by Go code from the policy model inputs currently implemented in GoMLX. The
parser and batcher preserve more fields than the policy model consumes.

## Parsed Sample

The parser reads binary format version `2` records into `data.Sample`. Each record is 234 bytes and uses the constants
`MaxTurns = 5`, `WordLength = 5`, and `TopK = 16`.

| Field | Go type | Shape / size | Meaning |
| --- | --- | ---: | --- |
| `SolutionWord` | `data.Word` (`[5]byte`) | `word[5]` | Hidden solution word used to generate the record. It is loaded for auditing and split/evaluation checks. It is not model input. The train opening-state record uses the empty all-zero word. |
| `TurnDepth` | `uint8` | scalar | Number of previous turns in the visible state. The train opening state has `0`; non-opening records use `1..5`. The policy model does not consume this as a scalar turn-number feature, but it is needed to decide which previous-turn slots are occupied and to derive the virgin-grid flag. |
| `PreviousGuessWords` | `[5]data.Word` | `[5,5]` bytes | Five fixed-width previous-guess slots in chronological order. Only slots before `TurnDepth` are occupied turns; later slots are all-zero padding. |
| `PreviousFeedback` | `[5][5]data.Feedback` | `[5,5]` bytes | Feedback for each previous guess position. The file stores used turns as `0` grey, `1` yellow, or `2` green. Unused turns contain padding value `255`. The policy input encoder remaps used feedback values into the model-spec order green/yellow/grey. |
| `ShortlistSizeBefore` | `uint16` | scalar | Number of solution words still compatible with the visible state before the teacher selects the next guess. This is parsed and batched for inspection or possible loss weighting, but the model input spec excludes candidate constraints computed from the grid, so this is not fed to the policy model. |
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
| `Batch.Inputs[].ShortlistSizeBefore` | `uint16` | `[B]` | Compatible shortlist size before the teacher guess. Preserved in the batch, but excluded from the policy model input. |
| `Batch.Targets[].TopKGuessWords` | `[16]data.Word` | `[B,16,5]` | Ranked teacher guess words. |
| `Batch.Targets[].TopKReductionRatios` | `[16]float32` | `[B,16]` | Ranked teacher scores. |
| `Batch.Targets[].TopKWorstCaseSizes` | `[16]uint16` | `[B,16]` | Ranked teacher worst-case sizes. |

`B` is the actual batch size. The sequential batch iterator includes the final partial batch, so `B` may be smaller than
the configured batch size for the last batch.

## Policy Model Input

The GoMLX policy input conversion follows `docs/model-input-spec.md`, which describes the state input contract.

The model consumes a Wordle decision state before the next guess. It includes:

- up to five previous guess words;
- tile feedback for each previous guess;
- one scalar virgin-grid flag.

The policy model input excludes:

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

### State Tensors

`training.BatchToPolicyStateTensors` converts one `data.Batch` into the three state tensors consumed by
`model.PolicyVector` and the first three inputs of `model.PolicyModel`:

| Tensor | DType | Shape | Source |
| --- | --- | ---: | --- |
| raw turn features | `float32` | `[B,5,145]` | One-hot previous guesses and remapped feedback for occupied turns. |
| occupied-turn mask | `float32` | `[B,5]` | `1.0` for turns before `TurnDepth`; `0.0` for empty slots. |
| virgin-grid flag | `float32` | `[B,1]` | `1.0` when `TurnDepth == 0`; otherwise `0.0`. |

The fourth `model.PolicyModel` input is fixed action features with shape `[action_count,26]`. That tensor belongs to the
output/action-embedding contract in `docs/model-architecture.md`; it is not part of the state input spec.

### Dense Trunk Input

Inside the GoMLX graph, occupied raw turns are passed through the shared turn encoder and empty slots are masked to
literal zero vectors. The graph then builds the 321-value dense-trunk input:

```text
1 virgin-grid flag + (5 turn slots * 64 encoded values) = 321 values
```

For a batch, the graph's trunk input shape is:

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

The dense trunk is:

```text
321 -> 256 -> 256 -> 128 -> 128 -> 64
```

Use ReLU after each 256-value layer and after each 128-value layer. Do not apply an activation to the final 64-value
vector.

## Policy Training

`actionspace.Load` imports the 4,739-word action vocabulary from
`github.com/sam-bee/wordle-ml_game-engine/words.GetActionSpace`. The loader preserves the game-engine order and builds a
`data.Word -> action_index` map.

`training.BatchToPolicyTensors` converts one `data.Batch` and an action vocabulary into:

| Tensor | DType | Shape | Source |
| --- | --- | ---: | --- |
| raw turn features | `float32` | `[B,5,145]` | `training.BatchToPolicyStateTensors` |
| occupied-turn mask | `float32` | `[B,5]` | `training.BatchToPolicyStateTensors` |
| virgin-grid flag | `float32` | `[B,1]` | `training.BatchToPolicyStateTensors` |
| fixed action features | `float32` | `[action_count,26]` | `model.FixedActionFeatureMatrix` over the action vocabulary |
| teacher top-k indices | `int32` | `[B,16]` | Each `TopKGuessWords` entry mapped through the action vocabulary |

The fixed action features are model inputs, and the teacher top-k indices are labels consumed by `training.PolicyLoss`.
If any teacher word is missing from the action vocabulary, conversion fails clearly.

`training.NewPolicyTrainer` builds one trainer and keeps the CUDA backend, GoMLX context, optimizer, model variables, and
action vocabulary alive across batches. `training.RunPolicyStep` remains as a focused one-batch smoke helper built on
the same trainer.

| Component | Current choice |
| --- | --- |
| backend | GoMLX XLA backend with CUDA PJRT plugin |
| model | `model.PolicyModel` |
| target | teacher top-16 action indices |
| loss | `training.PolicyLoss` from `docs/loss-shaping.md` |
| optimizer | SGD, initial learning rate `0.05` by default, optional GoMLX decay with `--learning-rate-decay` |
| execution | validation loss before training, one or more training epochs, validation loss after each epoch |

When `--learning-rate-decay` is enabled, GoMLX SGD uses `initial_learning_rate / sqrt(global_step)`. The first update
uses the configured initial learning rate. The trainer disables GoMLX PJRT auto-installation in code and requires the
CUDA backend to expose exactly one visible device. The system environment is responsible for masking CUDA visibility so
that this one device is the RTX 5070 Ti.
The CLI saves native GoMLX checkpoints after each completed epoch, writes a project manifest, and emits TensorBoard
scalar telemetry under the run directory. Saved standalone model export is still intentionally not implemented.
Checkpoint resume is opt-in with `--resume`; without it, the CLI starts a new checkpoint run directory.
