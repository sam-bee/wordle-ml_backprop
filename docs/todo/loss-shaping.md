# Supervised Policy Loss Contract

This document describes the agreed loss shaping for the first real supervised Wordle policy model.

## Goal

The model should learn to choose a good next guess globally, not merely rank the teacher's top-k guesses against each
other.

For each training sample, the teacher provides:

* the best next guess;
* the next 15 preferred guesses;
* optional per-record score information.

For the first real training objective, the teacher's best guess is the primary target. The rest of the teacher top-16
guesses are used as a softer auxiliary signal.

## Model Output

The model produces a policy latent vector for the current visible Wordle state:

```text
v = policy_latent(state)
```

Each possible guess word has an output embedding:

```text
emb(w)
```

The raw model score for a word is the dot product:

```text
score(w) = v · emb(w)
```

The model scores every allowed guess in the full action vocabulary.

## Confidence

Scores are converted into full-vocabulary confidence values using softmax:

```text
confidence(w_x) =
    exp(score(w_x))
    /
    sum over all allowed words w_i of exp(score(w_i))
```

Equivalently:

```text
confidence(w_x) =
    exp(v · emb(w_x))
    /
    sum over all allowed words w_i of exp(v · emb(w_i))
```

The word with the highest dot product also has the highest confidence.

## Teacher Targets

Let:

```text
t_1 = teacher's best guess
t_2 = teacher's second-best guess
...
t_16 = teacher's sixteenth-best guess
```

The main target is:

```text
t_1
```

The auxiliary teacher-good set is:

```text
{t_1, t_2, ..., t_16}
```

## Loss Shape

The total loss is:

```text
total_loss =
    α * main_loss
    +
    β * auxiliary_loss
```

where:

```text
α >= 0
β >= 0
α + β = 1
```

## Main Loss

The main loss should strongly punish the model when the teacher's best guess has low confidence.

Use negative log confidence:

```text
main_loss = -log(confidence(t_1))
```

This is equivalent to full-vocabulary cross-entropy with the teacher's best guess as the correct class.

Plain English:

```text
Make the teacher's #1 guess the model's global top choice.
```

## Auxiliary Loss

The auxiliary loss should encourage the model to place confidence on the teacher's top-16 guesses, with earlier teacher
guesses weighted more strongly.

A simple version is:

```text
auxiliary_loss =
    - sum from j=1 to 16 of q_j * log(confidence(t_j))
```

where `q_j` is a fixed teacher-rank weight.

For example:

```text
q_j = 2^-j / Z
```

where `Z` normalises the weights so that:

```text
sum from j=1 to 16 of q_j = 1
```

Plain English:

```text
Also reward the model for giving confidence to the other teacher-good guesses,
especially the higher-ranked ones.
```

## Recommended First Settings

Start simple:

```text
α = 0.8
β = 0.2
```

That means:

```text
80% of the training signal says:
  make teacher #1 the global answer.

20% of the training signal says:
  keep probability mass on the teacher top-16.
```

These values are not sacred. They are a starting point.

## What This Loss Does

This loss trains the model to:

* score every allowed guess in the full vocabulary;
* make the teacher's #1 guess globally high-confidence;
* avoid putting excessive confidence on words outside the teacher's top-16;
* use the teacher's ordering as a soft preference, not a hard ranking rule.

## What This Loss Does Not Directly Enforce

This loss does not directly enforce hard ranking constraints such as:

```text
model top-4 must all be inside teacher top-16
```

or:

```text
teacher top-16 must exactly equal model top-16 in the same order
```

Those are better treated as evaluation metrics.

## Evaluation Metrics

During validation/testing, report metrics such as:

```text
top-1 exact match:
  model #1 == teacher #1

top-1 inside teacher top-16:
  model #1 ∈ {teacher top-16}

top-4 inside teacher top-16:
  all of model top-4 are in {teacher top-16}

top-16 overlap:
  count of model top-16 guesses that appear in teacher top-16
```

A separate bounded ranking metric may also be reported, but it should not be the main backprop loss if it depends on
discrete rank positions after sorting.

## Trainable Parameters

Training should update all normal trainable parameters end-to-end:

* input encoder weights;
* dense trunk weights;
* policy head weights;
* trainable output embedding dimensions.

Fixed/non-trainable output embedding dimensions must remain constant.

