# Training Data Format

This repository stores Wordle imitation-learning data in fixed-width binary files under `data/`. The matching JSON files
are metadata sidecars for checking counts and provenance; the training path should parse the `.bin` files.

The current binary format is version `2`.

## Files

The dataset is split by hidden solution word:

| Split | Binary file | Sidecar | Split ID | Records | Split solutions |
| --- | --- | --- | ---: | ---: | ---: |
| Train | `data/train/wordle-train.bin` | `data/train/wordle-train.json` | `1` | `46176` | `1847` |
| Validation | `data/validation/wordle-validation.bin` | `data/validation/wordle-validation.json` | `2` | `5750` | `230` |
| Test | `data/test/wordle-test.bin` | `data/test/wordle-test.json` | `3` | `5800` | `232` |

The train file includes one global opening-state record. Validation and test do not include an opening-state record.

## Byte Order And Primitive Types

All multi-byte numeric values are little-endian.

| Type | Size | Meaning |
| --- | ---: | --- |
| `uint8` | 1 byte | Unsigned integer |
| `uint16` | 2 bytes | Unsigned integer |
| `uint32` | 4 bytes | Unsigned integer |
| `float32` | 4 bytes | IEEE-754 single-precision float |
| `word[5]` | 5 bytes | Uppercase ASCII Wordle word |

Real word fields contain exactly five uppercase ASCII bytes. Empty or unused word fields are five zero bytes (`0x00 0x00
0x00 0x00 0x00`). Real words are not null-terminated.

## File Header

Each `.bin` file starts with a 64-byte header.

| Offset | Size | Type | Field | Current value / meaning |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | bytes | `magic` | ASCII `WDIT` |
| 4 | 4 | `uint32` | `version` | `2` |
| 8 | 4 | `uint32` | `record_count` | Number of records following the header |
| 12 | 4 | `uint32` | `top_k` | `16` |
| 16 | 4 | `uint32` | `max_turns` | `5` |
| 20 | 4 | `uint32` | `guess_vocab_size` | `4739` |
| 24 | 4 | `uint32` | `solution_count` | Number of split solutions in this file |
| 28 | 4 | `uint32` | `split_id` | `1` train, `2` validation, `3` test |
| 32 | 32 | bytes | reserved | All zero bytes |

Validate `magic`, `version`, `top_k`, `max_turns`, and file size before parsing records. The expected file size is:

```text
64 + record_count * 234
```

## Record Layout

Every record is exactly 234 bytes. Records start immediately after the 64-byte header.

Record `n` starts at byte offset:

```text
64 + n * 234
```

where `n` is zero-based and must be less than `record_count`.

| Relative offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 5 | `word[5]` | `solution_word` |
| 5 | 1 | `uint8` | `turn_depth` |
| 6 | 25 | `word[5] * 5` | `previous_guess_words` |
| 31 | 25 | `uint8 * 5 * 5` | `previous_feedback` |
| 56 | 2 | `uint16` | `shortlist_size_before` |
| 58 | 80 | `word[5] * 16` | `top_k_guess_words` |
| 138 | 64 | `float32 * 16` | `top_k_reduction_ratios` |
| 202 | 32 | `uint16 * 16` | `top_k_worst_case_sizes` |

Array fields are written in rank or turn order, with no separators.

`previous_guess_words` contains five fixed-width word slots:

```text
slot 0: bytes 6..10
slot 1: bytes 11..15
slot 2: bytes 16..20
slot 3: bytes 21..25
slot 4: bytes 26..30
```

`previous_feedback` contains five turns of five feedback values:

```text
turn 0: bytes 31..35
turn 1: bytes 36..40
turn 2: bytes 41..45
turn 3: bytes 46..50
turn 4: bytes 51..55
```

`top_k_guess_words`, `top_k_reduction_ratios`, and `top_k_worst_case_sizes` are parallel arrays. Index `0` is teacher
rank 1, index `15` is teacher rank 16.

## Record Semantics

Each record describes one incomplete Wordle state and the teacher policy's ranked next guesses for that state.

`solution_word` is the hidden solution used to generate the record. It is stored for auditing, split checking, and
game-level evaluation. For supervised policy training, do not feed `solution_word` to a model that is supposed to
imitate a real player, because a real player does not know the hidden solution.

The train opening-state record has:

```text
solution_word = empty padded word
turn_depth = 0
previous_guess_words = all empty padded words
previous_feedback = all 255
shortlist_size_before = 2309
```

For non-opening records, `turn_depth` is in the range `1..5`. Only the first `turn_depth` entries of
`previous_guess_words` and `previous_feedback` are part of the visible Wordle state. Remaining previous-turn slots are
padding:

```text
previous_guess_words[i] = empty padded word
previous_feedback[i][j] = 255
```

`shortlist_size_before` is the number of valid solution words still compatible with the previous guesses and feedback
before the teacher chooses the next guess. It is derived from the visible state and wordlists.

## Feedback Values

Feedback values use this enum:

| Value | Meaning |
| ---: | --- |
| `0` | Grey / absent |
| `1` | Yellow / present in wrong position |
| `2` | Green / correct position |
| `255` | Padding for unused previous-turn slots |

For each used previous turn, read exactly five feedback values. For example, `[2, 0, 2, 0, 2]` means `G-G-G`.

## Teacher Labels

Each record stores the teacher's top 16 legal next guesses.

For each rank `i` from `0` to `15`:

```text
guess = top_k_guess_words[i]
score = top_k_reduction_ratios[i]
worst_case_size = top_k_worst_case_sizes[i]
```

The labels are sorted best-first by the teacher. The teacher optimises worst-case shortlist reduction. The stored score
is:

```text
1.0 - worst_case_size / shortlist_size_before
```

Higher scores are better. Scores are meaningful within one record; do not treat them as globally comparable across
unrelated records without per-record normalisation.

The current files always contain 16 teacher labels per record.

The current generator writes the train opening-state record first. Other records are grouped by the hidden solution's
external solution index, then by `turn_depth`, then by an internal history key. Training code should not need to depend
on record order.

## Sidecar JSON

Each `.bin` file has a matching `.json` sidecar with the same basename. The sidecar contains metadata only; it does not
contain the training records.

Important sidecar fields:

| Field | Meaning |
| --- | --- |
| `version` | Binary format version, currently `2` |
| `binary_file` | Basename of the matching `.bin` file |
| `record_count` | Must match the header |
| `header_size_bytes` | `64` |
| `record_size_bytes` | `234` |
| `top_k` | `16` |
| `max_turns` | `5` |
| `guess_vocab_size` | `4739` |
| `global_solution_vocab_size` | `2309` |
| `solution_count` | Number of hidden solution words assigned to the split |
| `solution_ids` | Sorted external solution-word indices for the split |
| `records_per_solution` | `25` |
| `records_per_depth` | `5` |
| `includes_opening_state` | `true` only for train |
| `wordlist_hash` | SHA-256 hash of the source wordlist contents |
| `seed` | Generation seed |

The binary records store words directly, not vocabulary IDs. `solution_ids` are still useful for checking the split
against the external solution CSV, but they are not needed to parse the binary records.

## Supervised Learning Use

For imitation learning, a typical training example is:

Input state:

```text
turn_depth
previous_guess_words[0:turn_depth]
previous_feedback[0:turn_depth]
optionally shortlist_size_before
```

Target:

```text
top_k_guess_words
top_k_reduction_ratios
top_k_worst_case_sizes
```

The hidden `solution_word` should normally be excluded from policy-model inputs. It can be used for evaluation,
debugging, and verifying that train, validation, and test splits are disjoint by solution.

## Human-Readable Inspection

There is a CLI tool that can render a binary file as JSON for inspection:

```text
go run . human-readable data/wordle-train.bin
```

The human-readable files are not the canonical training format; they are just a direct decoded view of the binary
records.

If you want access to the human-readable CLI tool, you have to stop and ask.
