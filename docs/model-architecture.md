# Wordle Policy Model Architecture

This document specifies the policy model implemented in the
`/mnt/internal-ssd-2/ai/neuroevolution-wordle/codebase` project closely enough
to reimplement it in another language or ML tool.

It covers the model, parameter layout, input encoder, dense trunk, output
embeddings, initialization, and inference behavior. It does not specify the
whole genetic algorithm or fitness loop except where those systems affect model
parameters.

All indices below are zero-based.

## Source References

Primary source files inspected:

- `docs/neural-net-design/neural-net-design.md`
- `src/wordle/word.hpp`
- `src/wordle/feedback.hpp`
- `src/wordle/wordle_grid.hpp`
- `src/model/input_encoder/encoder_spec.hpp`
- `src/model/input_encoder/turn_features.hpp`
- `src/model/input_encoder/shared_encoder.hpp`
- `src/model/model_input/model_input_spec.hpp`
- `src/model/model_input/wordle_grid_state.hpp`
- `src/model/dense_trunk/dense_trunk.hpp`
- `src/model/output_embedding/output_embedding.hpp`
- `src/model/initialization/parameter_initialization.hpp`
- `src/model/initialization/parameter_initialization.cpp`
- `src/genetic_algorithm/genome/dynamic_layout.hpp`
- `src/inference/dynamic_policy.hpp`
- `docs/genetic-algorithm/winner-artifacts.md`
- `docs/genetic-algorithm/output-embedding-recombination.md`

## Top-Level Shape

The model maps a non-terminal Wordle decision state to a 64-dimensional policy
vector, then scores each active action word by dot product against that word's
64-dimensional output embedding.

```text
WordleGrid decision state
-> virgin-grid flag + up to 5 previous turns
-> shared per-turn encoder, repeated over occupied turns
-> 321-value model input vector
-> dense trunk
-> 64-value policy vector
-> dot product with active action embeddings
-> best-scoring action word
```

There is no recurrence, attention, convolution, batch normalization, dropout,
softmax, or learned per-action output bias.

## Word And Feedback Conventions

Words are exactly 5 uppercase ASCII letters.

Letter indices:

- `A -> 0`
- `B -> 1`
- ...
- `Z -> 25`

Feedback values are encoded in this exact enum order:

- `green -> 0`
- `yellow -> 1`
- `grey -> 2`

The input `WordleGrid` stores the solution, previous turns, and turn count. The
solution word is not directly encoded as a model input. It only affects the
model through the feedback stored on previous turns.

The model accepts only valid, non-terminal decision states with at most 5
previous turns. A won board or a 6-turn finished board is rejected.

## Per-Turn Input Encoder

Each occupied turn is first converted to a 145-value binary float vector:

```text
[5 * 26 guess-letter one-hot values | 5 * 3 feedback one-hot values]
```

### Guess-Letter Features

There are 130 guess-letter features.

For position `pos` in `0..4` and letter index `letter` in `0..25`:

```text
guess_feature_index = pos * 26 + letter
```

Exactly one feature is set to `1.0` for each position. All other guess-letter
features are `0.0`.

### Feedback Features

There are 15 feedback features. They start after the 130 guess-letter features.

For position `pos` in `0..4` and feedback index `fb` in `0..2`:

```text
feedback_feature_index = 130 + pos * 3 + fb
```

Exactly one feedback feature is set to `1.0` for each position. All other
feedback features are `0.0`.

The full turn vector length is:

```text
5 * 26 + 5 * 3 = 145
```

### Shared Encoder Network

The same encoder parameters are reused for every occupied turn slot. There are
not five separate encoders.

Shape:

```text
145 -> 128 -> 64
```

Forward equations:

```text
hidden_pre[j] = encoder_b0[j] + sum_i encoder_w0[j, i] * turn_x[i]
hidden[j] = max(0, hidden_pre[j])
encoded_turn[k] = encoder_b1[k] + sum_j encoder_w1[k, j] * hidden[j]
```

Layer details:

| Layer | Weight Shape | Bias Shape | Activation |
| --- | ---: | ---: | --- |
| `input_encoder.input_to_hidden` | `[128][145]` | `[128]` | ReLU |
| `input_encoder.hidden_to_output` | `[64][128]` | `[64]` | none |

The 64-dimensional encoder output is linear. Negative values are allowed.

## Empty Turn Slots And Empty Grid Behavior

The model always builds 5 turn slots, but the shared encoder is run only for
occupied turns.

For an empty slot, the model writes a literal 64-dimensional zero vector:

```text
[0.0, 0.0, ..., 0.0]
```

This is not equivalent to passing an all-zero 145-vector through the shared
encoder, because the encoder biases are skipped. Empty slots cannot contribute
through encoder weights or encoder biases.

The model input vector is initialized to all zeros before occupied turns are
written.

The empty Wordle grid has a special signal:

```text
model_input[0] = 1.0
model_input[1..320] = 0.0
```

So the first move is controlled by the dense trunk's biases and the trunk
weights connected to the virgin-grid flag. The per-turn encoder contributes
nothing on an empty grid.

For any non-empty grid:

```text
model_input[0] = 0.0
```

Remaining unoccupied slots stay zero-filled.

## Model Input Vector

After turn encoding, the dense trunk input has length 321:

```text
1 virgin-grid scalar + 5 * 64 encoded-turn values = 321
```

Layout:

```text
model_input[0] = 1.0 if grid.turn_count == 0 else 0.0
model_input[1 + turn_index * 64 + feature_index] = encoded turn value
```

`turn_index` is chronological: turn 0 is the earliest guess, turn 4 is the
latest representable guess.

## Dense Trunk

The dense trunk maps the 321-value model input vector to a 64-dimensional policy
vector.

Shape:

```text
321 -> 256 -> 256 -> 128 -> 128 -> 128 -> 64
```

Forward equations:

```text
hidden0_pre[j] = trunk_b0[j] + sum_i trunk_w0[j, i] * model_input[i]
hidden0[j] = max(0, hidden0_pre[j])

hidden1_pre[k] = trunk_b1[k] + sum_j trunk_w1[k, j] * hidden0[j]
hidden1[k] = max(0, hidden1_pre[k])

hidden2_pre[m] = trunk_b2[m] + sum_k trunk_w2[m, k] * hidden1[k]
hidden2[m] = max(0, hidden2_pre[m])

hidden3_pre[n] = trunk_b3[n] + sum_m trunk_w3[n, m] * hidden2[m]
hidden3[n] = max(0, hidden3_pre[n])

hidden4_pre[q] = trunk_b4[q] + sum_n trunk_w4[q, n] * hidden3[n]
hidden4[q] = max(0, hidden4_pre[q])

policy[p] = trunk_b5[p] + sum_q trunk_w5[p, q] * hidden4[q]
```

Layer details:

| Layer | Weight Shape | Bias Shape | Activation |
| --- | ---: | ---: | --- |
| `dense_trunk.input_to_hidden0` | `[256][321]` | `[256]` | ReLU |
| `dense_trunk.hidden0_to_hidden1` | `[256][256]` | `[256]` | ReLU |
| `dense_trunk.hidden1_to_hidden2` | `[128][256]` | `[128]` | ReLU |
| `dense_trunk.hidden2_to_hidden3` | `[128][128]` | `[128]` | ReLU |
| `dense_trunk.hidden3_to_hidden4` | `[128][128]` | `[128]` | ReLU |
| `dense_trunk.hidden4_to_output` | `[64][128]` | `[64]` | none |

The 64-dimensional policy vector is linear. There is no output activation and
no normalization.

## Parameter Storage And Ordering

All trainable dense-network parameters are stored as fp16 (`common::Float16` in
the C++ code). Forward code converts each stored fp16 value to `float` before
using it in dense-layer accumulation.

For every dense layer, weights are stored row-major by output neuron:

```text
flat_weight_index = output_index * input_size + input_index
```

Each dense layer stores:

```text
weights, then biases
```

The policy model parameter order is:

1. `input_encoder.input_to_hidden.weights` - `128 * 145 = 18,560` fp16 values
2. `input_encoder.input_to_hidden.biases` - `128` fp16 values
3. `input_encoder.hidden_to_output.weights` - `64 * 128 = 8,192` fp16 values
4. `input_encoder.hidden_to_output.biases` - `64` fp16 values
5. `dense_trunk.input_to_hidden0.weights` - `256 * 321 = 82,176` fp16 values
6. `dense_trunk.input_to_hidden0.biases` - `256` fp16 values
7. `dense_trunk.hidden0_to_hidden1.weights` - `256 * 256 = 65,536` fp16 values
8. `dense_trunk.hidden0_to_hidden1.biases` - `256` fp16 values
9. `dense_trunk.hidden1_to_hidden2.weights` - `128 * 256 = 32,768` fp16 values
10. `dense_trunk.hidden1_to_hidden2.biases` - `128` fp16 values
11. `dense_trunk.hidden2_to_hidden3.weights` - `128 * 128 = 16,384` fp16 values
12. `dense_trunk.hidden2_to_hidden3.biases` - `128` fp16 values
13. `dense_trunk.hidden3_to_hidden4.weights` - `128 * 128 = 16,384` fp16 values
14. `dense_trunk.hidden3_to_hidden4.biases` - `128` fp16 values
15. `dense_trunk.hidden4_to_output.weights` - `64 * 128 = 8,192` fp16 values
16. `dense_trunk.hidden4_to_output.biases` - `64` fp16 values

Parameter counts:

| Group | fp16 Values | Bytes |
| --- | ---: | ---: |
| Shared input encoder | 26,944 | 53,888 |
| Dense trunk | 222,400 | 444,800 |
| Policy model total | 249,344 | 498,688 |

## Output Embeddings

The model does not have a dense output neuron per word. Instead, each active
action word has a 64-dimensional embedding, and the 64-dimensional policy vector
is scored against each embedding by dot product.

The full curated action catalog contains 4,739 words. Runtime artifacts may use
an active prefix smaller than 4,739. The active words in a winner artifact's
JSON sidecar are authoritative.

Each action embedding is:

```text
[26 fixed word features | 38 trainable tail features]
```

### Fixed Word Features

Dimensions `0..25` are fixed and not trainable. They are recomputed from the
action word itself.

For letter index `l`:

```text
count = number of times letter l appears in the action word
fixed_feature[l] = -1.0 if count == 0 else float(count)
```

Examples for `CRASS`:

- `A -> +1.0`
- `B -> -1.0`
- `C -> +1.0`
- `R -> +1.0`
- `S -> +2.0`

These fixed features are count-aware, not just present/absent.

### Trainable Tail Features

Dimensions `26..63` are trainable. There are 38 fp16 tail values per active
action word.

For an action at index `a` and tail feature `t` in `0..37`:

```text
embedding[26 + t] = float(trainable_tail[a][t])
```

Only this 38-dimensional tail is trainable per action word. The fixed 26
dimensions are never trained, mutated, recombined, or stored as trainable genome
data.

There is no trainable scalar bias per action.

## Action Scoring And Selection

For policy vector `p` and action word `w`:

```text
score(w) =
    sum_i=0..25  p[i]      * fixed_word_feature(w)[i]
  + sum_t=0..37  p[26 + t] * trainable_tail[w][t]
```

The chosen action is the valid active action with the highest score.

Sequential selection keeps the first action when scores tie because it updates
only on `score > best_score`. Dynamic CUDA inference also tie-breaks toward the
lower action index.

Dynamic inference masks words that have already been guessed on the current
grid before selecting the best action. The lower-level `TrySelectBestAction`
helper does not apply this repeated-guess mask.

## Winner Artifact Genome Layout

Winner artifacts save one binary genome payload plus a JSON metadata sidecar.

The JSON sidecar supplies:

- `action_count`
- `genome_byte_count`
- `action_space_words`

`action_space_words` is the authoritative active action list. `action_space_path`
is provenance only.

The binary genome payload stores:

```text
offset 0:
    PolicyModelParameters, 498,688 bytes

offset 498,688:
    trainable output-tail rows for active actions
    row 0: 38 fp16 values
    row 1: 38 fp16 values
    ...
    row action_count - 1: 38 fp16 values
```

Each tail row is `38 * 2 = 76` bytes.

Current stride formula, with the current fp16-only layout and 2-byte alignment:

```text
genome_byte_count = 498688 + action_count * 76
```

For the full 4,739-word catalog:

```text
tail values = 4739 * 38 = 180,082 fp16 values
tail bytes = 360,164
full genome bytes = 858,852
```

The C++ source computes this through `ComputeDynamicGenomeStrideBytes`, which
rounds for alignment. Under the current layout that rounding does not add extra
padding beyond the values listed above.

## Initialization

The default host initializer uses `std::mt19937` and samples from normal
distributions. The default device generation-0 initializer uses Philox/curand
normal draws with matching distribution parameters.

There is no bounded uniform initialization range. Dense weights and output-tail
values are sampled from Gaussian distributions before conversion to fp16, so
the mathematical distribution is unbounded. The stored values are limited only
by fp16 representation and any later implementation-specific handling of fp16
overflow.

Default config:

```text
dense_weight_gain = 1.0
output_embedding_tail_stddev = 0.05
```

Validation rules:

```text
dense_weight_gain > 0.0
output_embedding_tail_stddev >= 0.0
```

Dense-layer weights use He-normal initialization:

```text
stddev = dense_weight_gain * sqrt(2.0 / fan_in)
weight ~ Normal(mean = 0.0, stddev)
bias = 0.0
```

Default dense weight standard deviations:

| Layer | Fan-In | Default Stddev |
| --- | ---: | ---: |
| `input_encoder.input_to_hidden` | 145 | 0.1174440439 |
| `input_encoder.hidden_to_output` | 128 | 0.125 |
| `dense_trunk.input_to_hidden0` | 321 | 0.0789337038 |
| `dense_trunk.hidden0_to_hidden1` | 256 | 0.0883883476 |
| `dense_trunk.hidden1_to_hidden2` | 256 | 0.0883883476 |
| `dense_trunk.hidden2_to_hidden3` | 128 | 0.125 |
| `dense_trunk.hidden3_to_hidden4` | 128 | 0.125 |
| `dense_trunk.hidden4_to_output` | 128 | 0.125 |

Output embedding trainable tails at initial random creation use:

```text
tail_value ~ Normal(mean = 0.0, stddev = output_embedding_tail_stddev)
```

with default `stddev = 0.05`, then conversion to fp16.

## Output-Tail Growth And Evolution Notes

These details are not part of the static forward architecture, but they affect
the trainable tail values that a saved model may contain.

During normal mutation, dense parameters and output-tail values receive small
additive Gaussian drift when selected for mutation. Current CLI defaults are:

```text
mutation_probability = 0.0001
mutation_sigma = 0.02
output_tail_row_scale_mutation_probability = 0.0
```

So the normal CLI path does not apply whole-row tail magnitude scaling unless a
caller changes that configuration.

When the action-space prefix grows, the runtime can inject new output-tail rows
for the newly active words. The injection path:

1. Computes the median L2 norm of existing trainable tail rows.
2. Builds hint-grid decision states for each new target word.
3. Runs the current policy model on those hint-grid states.
4. Averages policy dimensions `26..63`.
5. Stores that 38-dimensional average as the new tail row.
6. Scales the new tail row to the median existing tail norm.

This only seeds the 38 trainable tail features. The 26 fixed word features
remain deterministic from the action word.

## Minimal Forward Pseudocode

```text
relu(x):
    return max(0.0, x)

dense(W, b, x):
    for out in 0..output_size-1:
        y[out] = float(b[out])
        for i in 0..input_size-1:
            y[out] += float(W[out][i]) * x[i]
    return y

encode_turn(turn):
    x = zeros(145)
    for pos in 0..4:
        x[pos * 26 + turn.guess.letter_indices[pos]] = 1.0
        fb = turn.feedback[pos]  // green=0, yellow=1, grey=2
        x[130 + pos * 3 + fb] = 1.0

    h = dense(encoder_w0, encoder_b0, x)
    for j in 0..127:
        h[j] = relu(h[j])
    return dense(encoder_w1, encoder_b1, h)

encode_grid(grid):
    reject if invalid, finished, or grid.turn_count > 5

    m = zeros(321)
    m[0] = 1.0 if grid.turn_count == 0 else 0.0

    for t in 0..grid.turn_count-1:
        e = encode_turn(grid.turns[t])
        for k in 0..63:
            m[1 + t * 64 + k] = e[k]

    return m

forward_policy(grid):
    m = encode_grid(grid)

    h0 = dense(trunk_w0, trunk_b0, m)
    for j in 0..255:
        h0[j] = relu(h0[j])

    h1 = dense(trunk_w1, trunk_b1, h0)
    for j in 0..255:
        h1[j] = relu(h1[j])

    h2 = dense(trunk_w2, trunk_b2, h1)
    for j in 0..127:
        h2[j] = relu(h2[j])

    h3 = dense(trunk_w3, trunk_b3, h2)
    for j in 0..127:
        h3[j] = relu(h3[j])

    h4 = dense(trunk_w4, trunk_b4, h3)
    for j in 0..127:
        h4[j] = relu(h4[j])

    return dense(trunk_w5, trunk_b5, h4)  // length 64

fixed_word_features(word):
    counts = zeros(26)
    for pos in 0..4:
        counts[word.letter_indices[pos]] += 1
    for l in 0..25:
        features[l] = -1.0 if counts[l] == 0 else float(counts[l])
    return features

score_action(policy, action_word, tail_row):
    fixed = fixed_word_features(action_word)
    score = 0.0
    for i in 0..25:
        score += policy[i] * fixed[i]
    for t in 0..37:
        score += policy[26 + t] * float(tail_row[t])
    return score

select_action(grid, action_words, tail_rows):
    policy = forward_policy(grid)
    best_index = none
    best_score = none

    for a in 0..action_count-1:
        if dynamic_inference and action_words[a] was already guessed:
            continue
        score = score_action(policy, action_words[a], tail_rows[a])
        if best_index is none or score > best_score:
            best_index = a
            best_score = score

    reject if no candidate remains
    return action_words[best_index], best_score, best_index
```
