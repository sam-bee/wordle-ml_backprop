# Model/Data Contract

This document describes the current data contract implemented in code. It is not the final Wordle model contract.

## Parsed Sample

The parser reads binary format version `2` records into `data.Sample`. Each record is 234 bytes and uses the constants
`MaxTurns = 5`, `WordLength = 5`, and `TopK = 16`.

| Field | Go type | Shape / size | Meaning |
| --- | --- | ---: | --- |
| `SolutionWord` | `data.Word` (`[5]byte`) | `word[5]` | Hidden solution word used to generate the record. It is loaded for auditing and split/evaluation checks. It is not included in `BatchInput` and should not be treated as player-visible model input. The train opening-state record uses the empty all-zero word. |
| `TurnDepth` | `uint8` | scalar | Number of previous turns in the visible state. The train opening state has `0`; non-opening records use `1..5`. |
| `PreviousGuessWords` | `[5]data.Word` | `[5,5]` bytes | Five fixed-width previous-guess slots. Only slots before `TurnDepth` are visible turns; later slots are all-zero padding. |
| `PreviousFeedback` | `[5][5]data.Feedback` | `[5,5]` bytes | Feedback for each previous guess position. Used turns contain `0` grey, `1` yellow, or `2` green. Unused turns contain padding value `255`. |
| `ShortlistSizeBefore` | `uint16` | scalar | Number of solution words still compatible with the visible state before the teacher selects the next guess. |
| `TopKGuessWords` | `[16]data.Word` | `[16,5]` bytes | Teacher's ranked legal next guesses, best first. |
| `TopKReductionRatios` | `[16]float32` | `[16]` | Teacher score per ranked guess. The file-format doc defines this as `1.0 - worst_case_size / shortlist_size_before`; higher is better within a record. |
| `TopKWorstCaseSizes` | `[16]uint16` | `[16]` | Worst-case remaining shortlist size for each ranked teacher guess. |

## Batch Contract

`data.BuildBatch` converts a slice of parsed samples into:

| Batch field | Go type | Batch shape | Meaning |
| --- | --- | ---: | --- |
| `Batch.Inputs[].TurnDepth` | `uint8` | `[B]` | Visible turn count for each sample. |
| `Batch.Inputs[].PreviousGuessWords` | `[5]data.Word` | `[B,5,5]` | Fixed previous-guess slots for each sample. |
| `Batch.Inputs[].PreviousFeedback` | `[5][5]data.Feedback` | `[B,5,5]` | Fixed feedback slots for each sample. |
| `Batch.Inputs[].ShortlistSizeBefore` | `uint16` | `[B]` | Compatible shortlist size before the teacher guess. |
| `Batch.Targets[].TopKGuessWords` | `[16]data.Word` | `[B,16,5]` | Ranked teacher guess words. |
| `Batch.Targets[].TopKReductionRatios` | `[16]float32` | `[B,16]` | Ranked teacher scores. |
| `Batch.Targets[].TopKWorstCaseSizes` | `[16]uint16` | `[B,16]` | Ranked teacher worst-case sizes. |

`B` is the actual batch size. The sequential batch iterator includes the final partial batch, so `B` may be smaller than
the configured batch size for the last batch.

## GoMLX Sanity Contract

The current GoMLX path is only a training sanity check.

`training.BatchToTensors` converts one `data.Batch` into:

| Tensor | DType | Shape | Source |
| --- | --- | ---: | --- |
| input | `float32` | `[B,52]` | Temporary flattened numeric features. |
| label | `float32` | `[B,16]` | `TopKReductionRatios` only. |

The input feature order is:

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

## Current Mismatches And Open Questions

No parser/file-format mismatch is currently known: the parser constants and loaded fields match
`docs/training-data-format.md`.

Open questions before the real model:

- The real target must be chosen deliberately. The parsed teacher target includes guess words, reduction ratios, and
  worst-case sizes; the sanity model trains only on reduction ratios.
- `TopKGuessWords` are stored as words, not vocabulary IDs. A real policy model likely needs a deliberate guess-vocab
  target encoding.
- `SolutionWord` is parsed but excluded from batches and sanity-model inputs. It should remain excluded from player-like
  policy inputs unless a future evaluation/debug path explicitly needs it.
- The sanity encoding includes all five previous-turn slots and relies on padding values rather than slicing by
  `TurnDepth`. The real model needs an explicit decision about padding, masks, or variable-depth handling.
- The sanity feedback encoding collapses grey and padding to the same numeric value `0`. The real model should preserve
  that distinction or use masks if it matters.
- The current word-byte encoding is an ASCII-normalized placeholder. The real model needs a deliberate character,
  word-ID, or embedding strategy.
- `ShortlistSizeBefore` is included as an input feature in batching and the sanity tensor conversion, but the final
  architecture still needs to decide whether it is allowed as an input.
