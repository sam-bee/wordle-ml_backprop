# Wordle Policy Model Input Specification

This document specifies the input expected by the Wordle policy model. It is written as an implementation-independent
contract for reimplementing the same model architecture in another language.

The Go implementation currently lives in `training.BatchToPolicyStateTensors`, which builds the raw turn features,
occupied-turn mask, and virgin-grid flag consumed by `model.PolicyVector` and `model.PolicyModel`.

## Scope

The model consumes a Wordle decision state and produces a policy vector for choosing the next guess. The decision state is
the state before a guess is made.

The model input includes:

- up to 5 previous guess words
- tile feedback for each previous guess
- one scalar flag indicating whether there are no previous guesses

The model input does not include:

- the hidden solution word
- candidate-word constraints computed from the grid
- a recurrent hidden state
- an attention mask
- a separate turn number feature, other than slot order and the virgin-grid flag

## Valid Decision States

Represent at most 5 previous turns. Each previous turn has:

- a 5-letter guess
- 5 tile feedback values for that guess

The 5-turn limit exists because the model is used to choose the next guess. A state with 5 previous non-winning turns is
valid because it is the decision state before the sixth guess. Terminal states should not be encoded:

- no state after 6 guesses
- no state after the game has already been won

The solution word may be used by the game engine to generate feedback, but it is not part of the model input.

## Word And Feedback Conventions

Letters are encoded using uppercase English alphabet indices:

```text
A -> 0
B -> 1
...
Z -> 25
```

Positions are zero-based and run left to right:

```text
position 0, position 1, position 2, position 3, position 4
```

Feedback has 3 possible values, encoded in this order:

```text
green  -> 0
yellow -> 1
grey   -> 2
```

If using symbols, the project convention is:

```text
G -> green
Y -> yellow
- -> grey
```

## Per-Turn Raw Feature Vector

Each occupied turn is first converted to a 145-value binary vector:

```text
5 letter positions * 26 letters = 130 values
5 feedback positions * 3 colours = 15 values
total = 145 values
```

The layout is:

```text
[letter one-hot features | feedback one-hot features]
```

Letter features occupy indices `0..129`.

For a letter at position `p` with alphabet index `l`:

```text
letter_feature_index = (p * 26) + l
```

Feedback features occupy indices `130..144`.

For feedback at position `p` with feedback index `f`:

```text
feedback_feature_index = 130 + (p * 3) + f
```

All values start at `0.0`. For each occupied turn, exactly 10 values are set to `1.0`: one letter feature and one
feedback feature for each of the 5 positions.

Example for guess `ABCDE` with feedback `G Y - G Y`:

```text
active letter indices:   0, 27, 54, 81, 108
active feedback indices: 130, 134, 138, 139, 143
```

## Shared Turn Encoder

Each occupied turn's 145-value raw feature vector is passed through the same shared turn encoder:

```text
145 -> 128 -> 64
```

Use ReLU after the 128-value hidden layer. Do not apply an activation to the 64-value encoder output.

The same encoder weights are reused for every occupied turn slot. There are not five independent turn encoders.

## Empty Turn Slots

The model always has 5 chronological turn slots.

For each slot:

- if the slot contains a previous turn, encode that turn with the shared turn encoder to get a 64-value vector
- if the slot is empty, use a literal 64-value zero vector

Important: do not encode an all-zero 145-value vector for empty turns. Empty slots bypass the shared encoder and directly
contribute 64 zeroes.

## Full Model Input Vector

After turn encoding, construct a 321-value vector for the dense trunk:

```text
1 virgin-grid flag + (5 turn slots * 64 encoded values) = 321 values
```

Index `0` is the virgin-grid flag:

```text
input[0] = 1.0 if there are zero previous turns
input[0] = 0.0 otherwise
```

The 5 encoded turn slots follow in chronological order, oldest first:

```text
slot 0 starts at input[1]
slot 1 starts at input[65]
slot 2 starts at input[129]
slot 3 starts at input[193]
slot 4 starts at input[257]
```

For slot `s` and encoded-turn value index `i`:

```text
input[1 + (s * 64) + i] = encoded_turn_s[i]
```

where `s` is in `0..4` and `i` is in `0..63`.

If a slot is empty, all 64 values for that slot are `0.0`.

## Full Encoding Pseudocode

```text
function encode_model_input(previous_turns):
    assert 0 <= len(previous_turns) <= 5
    assert state is not terminal

    input = array_of_float(321, fill = 0.0)
    input[0] = 1.0 if len(previous_turns) == 0 else 0.0

    for slot_index from 0 to len(previous_turns) - 1:
        turn = previous_turns[slot_index]

        raw = array_of_float(145, fill = 0.0)

        for position from 0 to 4:
            letter_index = alphabet_index(turn.guess[position])
            feedback_index = feedback_value_index(turn.feedback[position])

            raw[(position * 26) + letter_index] = 1.0
            raw[130 + (position * 3) + feedback_index] = 1.0

        encoded = shared_turn_encoder(raw)  # length 64

        base = 1 + (slot_index * 64)
        for i from 0 to 63:
            input[base + i] = encoded[i]

    return input
```

For a completely empty grid, the final 321-value vector is:

```text
input[0] = 1.0
input[1..320] = 0.0
```

For any non-empty grid:

```text
input[0] = 0.0
occupied slots contain shared-encoder outputs
remaining future slots contain zeroes
```

## Dense Trunk Input

The 321-value vector above is the direct input to the dense trunk:

```text
321 -> 256 -> 128 -> 64
```

Use ReLU after the 256-value hidden layer and after the 128-value hidden layer. Do not apply an activation to the final
64-value policy vector.
