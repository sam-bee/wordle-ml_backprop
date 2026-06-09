# Random Model Initialization Specification

This document specifies how this project initializes a new randomized Wordle policy model.

## Scope

A randomized model consists of:

- policy-model dense parameters
- trainable output-embedding tail rows for the active action words

The action words themselves are not random model parameters. They come from the action catalog provided by
`github.com/sam-bee/wordle-ml_game-engine/words.GetActionSpace`, with capacity 4,739 words. A randomized model is
parameterized by an active action count `A`, where:

```text
1 <= A <= 4739
```

The default active action count is `20`, but the initialization rule is the same for any valid `A`.

## Numeric Representation

GoMLX initializes trainable variables in the graph variable dtype. The current training path supplies `float32` tensors,
so current policy variables are `float32`.

There is no clipping or hard min/max range before assignment. The normal distributions are unbounded.

## Initialization Parameters

Default initialization configuration:

```text
dense_weight_gain = 1.0
output_embedding_tail_stddev = 0.05
```

Valid configuration ranges:

```text
dense_weight_gain > 0.0
output_embedding_tail_stddev >= 0.0
```

If `output_embedding_tail_stddev` is `0.0`, every randomly initialized trainable tail value is exactly zero after
sampling.

## Dense Layer Initialization

Every dense layer uses the same rule:

```text
weight ~ Normal(mean = 0.0, stddev = dense_weight_gain * sqrt(2 / fan_in))
bias = 0.0
```

Each weight is sampled independently. Every bias is exactly zero.

GoMLX dense weights are conceptually indexed as:

```text
weights[input_index][output_neuron]
```

## Policy Model Layers

Initialize these layers in the policy model:

| Component | Layer | Shape | Fan-in | Weight stddev with default gain | Bias count |
| --- | --- | ---: | ---: | ---: | ---: |
| Shared turn encoder | input to hidden | `145 -> 128` | `145` | `0.1174440439` | `128` |
| Shared turn encoder | hidden to output | `128 -> 64` | `128` | `0.1250000000` | `64` |
| Dense trunk | input to hidden 0 | `321 -> 256` | `321` | `0.0789337038` | `256` |
| Dense trunk | hidden 0 to hidden 1 | `256 -> 256` | `256` | `0.0883883476` | `256` |
| Dense trunk | hidden 1 to hidden 2 | `256 -> 128` | `256` | `0.0883883476` | `128` |
| Dense trunk | hidden 2 to hidden 3 | `128 -> 128` | `128` | `0.1250000000` | `128` |
| Dense trunk | hidden 3 to hidden 4 | `128 -> 128` | `128` | `0.1250000000` | `128` |
| Dense trunk | hidden 4 to output | `128 -> 64` | `128` | `0.1250000000` | `64` |

Total policy-model trainable scalars:

```text
weights = 248,192
biases  = 1,152
total   = 249,344
```

All 248,192 weights are random normal samples. All 1,152 biases are zero.

## Output Embedding Structure

Each action word has a 64-value output embedding:

```text
64 = 26 fixed word features + 38 trainable tail features
```

The first 26 values are fixed features computed from the action word. They are not random and are not trainable.

For each letter index `l` from `0` to `25`, count how many times that letter appears in the 5-letter action word:

```text
if count(letter l) == 0: fixed_feature[l] = -1.0
otherwise:              fixed_feature[l] = float(count(letter l))
```

So fixed word features can be:

```text
-1.0, 1.0, 2.0, 3.0, 4.0, or 5.0
```

The remaining 38 values are trainable tail parameters.

For each active action row `a` in `0..A-1` and each trainable feature `j` in `0..37`:

```text
tail[a][j] ~ Normal(mean = 0.0, stddev = output_embedding_tail_stddev)
```

With default configuration:

```text
tail[a][j] ~ Normal(mean = 0.0, stddev = 0.05)
```

Each tail value is independently sampled.

There are:

```text
38 * A
```

random output-tail parameters.

## Action Scoring

The randomized tail values matter because action selection scores each candidate action by dot product with the
64-value policy vector:

```text
score(action) =
    sum i=0..25  policy[i]      * fixed_word_feature[action][i]
  + sum j=0..37  policy[26 + j] * tail[action][j]
```

The selected action is the active action word with the highest score.

## Output Embedding Magnitude Standardization

Initial random model creation does not standardize or normalize output embedding vectors.

Specifically, during initial random creation:

- fixed 26-value word features are used exactly as described above
- trainable 38-value tails are sampled from `Normal(0.0, 0.05)`
- no tail row is rescaled to a target norm
- no full 64-value output embedding is normalized
- no output embedding is clipped to a magnitude range

The expected trainable-tail L2 norm at initialization is approximately:

```text
sqrt(38) * 0.05 = 0.3082207001
```

This is only a distributional expectation, not an enforced value.

## Later Action-Space Changes

Changing the active action count after initial model creation is not implemented in this project yet. A future
implementation that adds new action rows should make a deliberate decision about how to initialize the new
output-embedding tail rows.

One possible approach is magnitude matching against existing tail rows:

1. Compute the L2 norm of each existing trainable tail row:

   ```text
   norm(row) = sqrt(sum j=0..37 tail[row][j]^2)
   ```

2. Compute the median of those existing row norms. If the row count is even, use the average of the two middle values.
   This median is the target norm.

3. For each newly injected action word, seed a raw 38-value tail from the current model's policy outputs on hint grids
   for that word:

   ```text
   raw_tail[j] = average over 3 hint grids of policy_output[26 + j]
   ```

   If the 3 hint grids cannot be built for the word, injection fails.

4. Let `raw_norm` be the norm of the raw tail row.

5. If the target norm is greater than `1.0e-6`, require `raw_norm > 1.0e-6` and scale:

   ```text
   scaled_tail[j] = raw_tail[j] * (target_norm / raw_norm)
   ```

6. If the target norm is less than or equal to `1.0e-6`, scale the row to all zeroes.

Only the 38-value trainable tail is norm-matched. The fixed 26-value word-feature prefix is not included in this norm
and is not rescaled.

## Creation Pseudocode

```text
function make_random_model(action_words, action_count, rng, config):
    assert 1 <= action_count <= len(action_words)
    assert config.dense_weight_gain > 0.0
    assert config.output_embedding_tail_stddev >= 0.0

    model = empty_model()

    initialize_dense_layer(model.encoder_input_to_hidden, 145, 128, rng, config.dense_weight_gain)
    initialize_dense_layer(model.encoder_hidden_to_output, 128, 64, rng, config.dense_weight_gain)
    initialize_dense_layer(model.trunk_input_to_hidden0, 321, 256, rng, config.dense_weight_gain)
    initialize_dense_layer(model.trunk_hidden0_to_hidden1, 256, 256, rng, config.dense_weight_gain)
    initialize_dense_layer(model.trunk_hidden1_to_hidden2, 256, 128, rng, config.dense_weight_gain)
    initialize_dense_layer(model.trunk_hidden2_to_hidden3, 128, 128, rng, config.dense_weight_gain)
    initialize_dense_layer(model.trunk_hidden3_to_hidden4, 128, 128, rng, config.dense_weight_gain)
    initialize_dense_layer(model.trunk_hidden4_to_output, 128, 64, rng, config.dense_weight_gain)

    for action_index from 0 to action_count - 1:
        model.action_words[action_index] = action_words[action_index]
        for feature_index from 0 to 37:
            value = normal_sample(rng, mean = 0.0, stddev = config.output_embedding_tail_stddev)
            model.tail[action_index][feature_index] = value

    return model

function initialize_dense_layer(layer, input_size, output_size, rng, dense_weight_gain):
    stddev = dense_weight_gain * sqrt(2.0 / input_size)

    for input_index from 0 to input_size - 1:
        for output_index from 0 to output_size - 1:
            value = normal_sample(rng, mean = 0.0, stddev = stddev)
            layer.weights[input_index][output_index] = value

    for output_index from 0 to output_size - 1:
        layer.biases[output_index] = 0.0
```

The exact RNG implementation does not affect the model contract unless bit-for-bit seed compatibility is required. The
important rule is independent normal draws with the means, standard deviations, dimensions, and initialization rules
specified above.

## GoMLX Implementation Status

The GoMLX implementation exposes these rules in `internal/model`:

- `RandomInitializationConfig` holds `dense_weight_gain` and `output_embedding_tail_stddev`.
- `ConfigureRandomInitialization` stores validated initialization settings on a GoMLX context.
- Dense layers in `PolicyModel` use `Normal(0, dense_weight_gain * sqrt(2/fan_in))` weights and zero biases.
- The trainable output tail uses `Normal(0, output_embedding_tail_stddev)`.
- `FixedActionFeatures` and `FixedActionFeatureMatrix` build the fixed 26-value action feature prefix.

The current project persists model state through native GoMLX checkpoints, so GoMLX variables are initialized and saved in
the graph variable dtype.
