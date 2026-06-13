# Wordle Policy Model Architecture

This document describes the current Wordle policy model implemented by
`internal/model`. It is the architecture contract for this GoMLX backprop
project.

The model maps a non-terminal Wordle decision state to one logit per active
action word. Training then compares those logits with the supervised teacher
targets described in `docs/loss-shaping.md`.

Related documents:

- `docs/model-input-spec.md` describes the state input encoding.
- `docs/model-contract.md` describes parsed samples, batches, tensors, and the
  training contract.
- `docs/random-model-initialisation-spec.md` describes current initialization
  rules.

All indices below are zero-based.

## Top-Level Shape

The model maps a Wordle decision state to a 48-dimensional policy vector, then
scores each action word by dot product against that word's 48-dimensional output
embedding.

```text
Wordle decision state
-> virgin-grid flag + up to 5 previous turns
-> shared per-turn encoder, repeated over occupied turns
-> 321-value dense-trunk input vector
-> dense trunk
-> 64-value trunk output
-> 48-value policy vector
-> dot product with action embeddings
-> action logits
```

There is no recurrence, attention, convolution, batch normalization, dropout,
softmax, or learned per-action scalar bias.

## Word And Feedback Conventions

Words are exactly 5 uppercase ASCII letters.

Letter indices:

- `A -> 0`
- `B -> 1`
- ...
- `Z -> 25`

Feedback values use this model order:

- `green -> 0`
- `yellow -> 1`
- `grey -> 2`

The model consumes only visible decision-state information. The solution word is
not a model input. It can affect the model only indirectly through previous-turn
feedback.

The model accepts valid, non-terminal decision states with at most five previous
turns. A won board or a six-turn finished board should not be sent to the policy
model for next-action selection.

## GoMLX Tensor Inputs

`model.PolicyModel` expects four tensors:

| Tensor | DType | Shape | Meaning |
| --- | --- | ---: | --- |
| raw turn features | `float32` | `[B,5,145]` | One-hot previous guesses and feedback for each possible turn slot. |
| occupied-turn mask | `float32` | `[B,5]` | `1.0` for occupied previous-turn slots, `0.0` for empty slots. |
| virgin-grid flag | `float32` | `[B,1]` | `1.0` only before the first guess, otherwise `0.0`. |
| fixed action features | `float32` | `[A,26]` | Deterministic word-count features for the active action words. |

`B` is the batch size. `A` is the active action count, currently 4,739 for the
full action catalog loaded from the game-engine dependency.

`model.PolicyModel` returns:

| Tensor | DType | Shape | Meaning |
| --- | --- | ---: | --- |
| action logits | `float32` | `[B,A]` | Unnormalized score for each active action word. |

## Per-Turn Input Encoder

Each occupied turn is represented as a 145-value binary vector:

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
hidden_pre[j] = encoder_b0[j] + sum_i encoder_w0[i, j] * turn_x[i]
hidden[j] = max(0, hidden_pre[j])
encoded_turn[k] = encoder_b1[k] + sum_j encoder_w1[j, k] * hidden[j]
```

Layer details:

| Layer | Weight Shape | Bias Shape | Activation |
| --- | ---: | ---: | --- |
| `input_encoder.input_to_hidden` | `[145,128]` | `[128]` | ReLU |
| `input_encoder.hidden_to_output` | `[128,64]` | `[64]` | none |

The 64-dimensional encoder output is linear. Negative values are allowed.

## Empty Turn Slots And Empty Grid Behavior

The model always represents five turn slots, but the shared encoder contributes
only for occupied turns.

For an empty slot, the model uses a literal 64-dimensional zero vector:

```text
[0.0, 0.0, ..., 0.0]
```

This is intentionally different from passing an all-zero 145-vector through the
shared encoder, because encoder biases must not affect empty slots.

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

## Dense Trunk Input

After turn encoding, the dense trunk input has length 321:

```text
1 virgin-grid scalar + 5 * 64 encoded-turn values = 321
```

Layout:

```text
model_input[0] = 1.0 if turn_count == 0 else 0.0
model_input[1 + turn_index * 64 + feature_index] = encoded turn value
```

`turn_index` is chronological: turn 0 is the earliest guess, turn 4 is the
latest representable guess.

## Dense Trunk

The dense trunk maps the 321-value input vector to a 64-dimensional trunk output.
A separate policy output head maps that trunk output to the 48-dimensional policy
vector.

Shape:

```text
321 -> 256 -> 128 -> 128 -> 64 -> 48
```

Forward equations:

```text
hidden0_pre[j] = trunk_b0[j] + sum_i trunk_w0[i, j] * model_input[i]
hidden0[j] = max(0, hidden0_pre[j])

hidden1_pre[k] = trunk_b1[k] + sum_j trunk_w1[j, k] * hidden0[j]
hidden1[k] = max(0, hidden1_pre[k])

hidden2_pre[m] = trunk_b2[m] + sum_k trunk_w2[k, m] * hidden1[k]
hidden2[m] = max(0, hidden2_pre[m])

trunk_output[p] = trunk_b3[p] + sum_m trunk_w3[m, p] * hidden2[m]
policy[q] = head_b[q] + sum_p head_w[p, q] * trunk_output[p]
```

Layer details:

| Layer | Weight Shape | Bias Shape | Activation |
| --- | ---: | ---: | --- |
| `dense_trunk.input_to_hidden0` | `[321,256]` | `[256]` | ReLU |
| `dense_trunk.hidden0_to_hidden1` | `[256,128]` | `[128]` | ReLU |
| `dense_trunk.hidden1_to_hidden2` | `[128,128]` | `[128]` | ReLU |
| `dense_trunk.hidden2_to_output` | `[128,64]` | `[64]` | none |
| `policy_output_head.trunk_to_policy` | `[64,48]` | `[48]` | none |

The 64-dimensional trunk output and 48-dimensional policy vector are linear.
There is no output activation and no normalization.

## Trainable Variables

GoMLX `layers.Dense` stores dense weights with shape `[input_dim, output_dim]`
and biases with shape `[output_dim]`. The current trainer supplies `float32`
input tensors, so the model variables are currently `float32`.

Dense variable scopes:

| Variable | Shape | Trainable Scalars |
| --- | ---: | ---: |
| `policy_model.input_encoder.input_to_hidden.dense.weights` | `[145,128]` | 18,560 |
| `policy_model.input_encoder.input_to_hidden.dense.biases` | `[128]` | 128 |
| `policy_model.input_encoder.hidden_to_output.dense.weights` | `[128,64]` | 8,192 |
| `policy_model.input_encoder.hidden_to_output.dense.biases` | `[64]` | 64 |
| `policy_model.dense_trunk.input_to_hidden0.dense.weights` | `[321,256]` | 82,176 |
| `policy_model.dense_trunk.input_to_hidden0.dense.biases` | `[256]` | 256 |
| `policy_model.dense_trunk.hidden0_to_hidden1.dense.weights` | `[256,128]` | 32,768 |
| `policy_model.dense_trunk.hidden0_to_hidden1.dense.biases` | `[128]` | 128 |
| `policy_model.dense_trunk.hidden1_to_hidden2.dense.weights` | `[128,128]` | 16,384 |
| `policy_model.dense_trunk.hidden1_to_hidden2.dense.biases` | `[128]` | 128 |
| `policy_model.dense_trunk.hidden2_to_output.dense.weights` | `[128,64]` | 8,192 |
| `policy_model.dense_trunk.hidden2_to_output.dense.biases` | `[64]` | 64 |
| `policy_model.policy_output_head.trunk_to_policy.dense.weights` | `[64,48]` | 3,072 |
| `policy_model.policy_output_head.trunk_to_policy.dense.biases` | `[48]` | 48 |

Dense parameter counts:

| Group | Trainable Scalars |
| --- | ---: |
| Shared input encoder | 26,944 |
| Dense trunk | 140,096 |
| Policy output head | 3,120 |
| Dense policy network total | 170,160 |

The output-embedding tail is also trainable:

| Variable | Shape | Trainable Scalars |
| --- | ---: | ---: |
| `output_embeddings.trainable_tail` | `[A,22]` | `A * 22` |

For the full 4,739-word action catalog, the tail contains 104,258 trainable
scalars and the whole policy model contains 274,418 trainable scalars.

Model persistence is handled by GoMLX checkpoints. The architecture contract
does not define a separate binary export format.

## Output Embeddings

The model does not have a dense output neuron per word. Instead, each active
action word has a 48-dimensional embedding, and the 48-dimensional policy vector
is scored against each embedding by dot product.

Each action embedding is:

```text
[26 fixed word features | 22 trainable tail features]
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

Dimensions `26..47` are trainable. There are 22 tail values per active action
word.

For an action at index `a` and tail feature `t` in `0..21`:

```text
embedding[26 + t] = trainable_tail[a][t]
```

Only this 22-dimensional tail is trainable per action word. The fixed 26
dimensions are deterministic features. There is no trainable scalar bias per
action.

## Action Scoring And Selection

For policy vector `p` and action word `w`:

```text
score(w) =
    sum_i=0..25  p[i]      * fixed_word_feature(w)[i]
  + sum_t=0..21  p[26 + t] * trainable_tail[w][t]
```

`model.PolicyModel` returns all action scores as logits with shape `[B,A]`.
Training uses those logits directly in the supervised policy loss.

Interactive play should ignore action words that have already been guessed on
the current grid before choosing the highest-scoring valid action. If two
remaining actions have exactly the same score, sequential selection keeps the
lower action index.

## Initialization

Dense weights use He-normal initialization:

```text
stddev = dense_weight_gain * sqrt(2.0 / fan_in)
weight ~ Normal(mean = 0.0, stddev)
bias = 0.0
```

Default configuration:

```text
dense_weight_gain = 1.0
output_embedding_tail_stddev = 0.05
```

Validation rules:

```text
dense_weight_gain > 0.0
output_embedding_tail_stddev >= 0.0
```

Default dense weight standard deviations:

| Layer | Fan-In | Default Stddev |
| --- | ---: | ---: |
| `input_encoder.input_to_hidden` | 145 | 0.1174440439 |
| `input_encoder.hidden_to_output` | 128 | 0.125 |
| `dense_trunk.input_to_hidden0` | 321 | 0.0789337038 |
| `dense_trunk.hidden0_to_hidden1` | 256 | 0.0883883476 |
| `dense_trunk.hidden1_to_hidden2` | 128 | 0.125 |
| `dense_trunk.hidden2_to_output` | 128 | 0.125 |
| `policy_output_head.trunk_to_policy` | 64 | 0.1767766953 |

Output embedding tail values use:

```text
trainable_tail[a][t] ~ Normal(mean = 0.0, stddev = output_embedding_tail_stddev)
```

with default `stddev = 0.05`.

## Minimal Forward Pseudocode

```text
relu(x):
    return max(0.0, x)

dense(W, b, x):
    for out in 0..output_size-1:
        y[out] = b[out]
        for i in 0..input_size-1:
            y[out] += x[i] * W[i][out]
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

encode_state(state):
    reject if invalid, finished, or state.turn_count > 5

    m = zeros(321)
    m[0] = 1.0 if state.turn_count == 0 else 0.0

    for t in 0..state.turn_count-1:
        e = encode_turn(state.turns[t])
        for k in 0..63:
            m[1 + t * 64 + k] = e[k]

    return m

forward_policy(state):
    m = encode_state(state)

    h0 = dense(trunk_w0, trunk_b0, m)
    for j in 0..255:
        h0[j] = relu(h0[j])

    h1 = dense(trunk_w1, trunk_b1, h0)
    for j in 0..127:
        h1[j] = relu(h1[j])

    h2 = dense(trunk_w2, trunk_b2, h1)
    for j in 0..127:
        h2[j] = relu(h2[j])

    trunk_output = dense(trunk_w3, trunk_b3, h2)  // length 64
    return dense(head_w, head_b, trunk_output)    // length 48

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
    for t in 0..21:
        score += policy[26 + t] * tail_row[t]
    return score

select_action(state, action_words, tail_rows):
    policy = forward_policy(state)
    best_index = none
    best_score = none

    for a in 0..action_count-1:
        if action_words[a] was already guessed in state:
            continue
        score = score_action(policy, action_words[a], tail_rows[a])
        if best_index is none or score > best_score:
            best_index = a
            best_score = score

    reject if no candidate remains
    return action_words[best_index], best_score, best_index
```
