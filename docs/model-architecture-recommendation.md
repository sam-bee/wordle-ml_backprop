# Model Architecture Recommendation

Date: 2026-06-15

## Recommendation

The strongest mini result so far is the wider dense-trunk architecture from branch
`experiment/mini-deeper-dense-trunk-wide`:

```text
shared turn encoder: 145 -> 128 -> 64
dense trunk:         321 -> 256 -> 256 -> 256 -> 128 -> 128 -> 128 -> 64
output embedding:   26 fixed letter-count features + 38 trainable tail features
```

Use this as the lead candidate for the next larger-data validation step, but keep
`experiment/mini-depth-plus-three` as the main challenger. The wider trunk has the best five-seed average final
validation loss, but its lead over `experiment/mini-depth-plus-three` is narrow: `7.812392` vs `7.824779`. On paired
seeds, the wider trunk won 2 of 5 seeds and averaged `0.012387` lower final validation loss.

Do not adopt the smaller output tail, flat raw-state encoder, transformer state encoder, or direct action-logit head
yet. They do not beat the best dense-trunk variants on final validation loss.

## Why This Direction

The current model already uses ReLU layers with He-style initialization. He et al. derived this initialization to keep
deep rectifier networks trainable from scratch, which makes a moderate-depth MLP trunk a reasonable first capacity
increase for this codebase: https://arxiv.org/abs/1502.01852

Depth should still be treated cautiously. ResNet showed that substantially deeper plain networks can become harder to
optimize and that residual parameterization helps depth scale: https://arxiv.org/abs/1512.03385. The five-seed rerun
supports a slightly wider/deeper plain trunk on the mini dataset, but the lead over the previous +3 trunk is small. If
we go materially deeper than the current wider trunk, the next architecture change should add residual blocks rather
than simply stacking more dense layers.

The flat-state encoder experiment tested early cross-turn mixing by removing independent per-turn compression. It did
not win. For this small, fixed-length chronological input, the existing shared turn encoder plus ordered concatenation
appears to be a better inductive bias than a single raw flat state vector. Attention remains a plausible future
direction for richer sequence mixing, but the Transformer motivation is strongest when each position needs to attend to
other positions through learned interactions: https://arxiv.org/abs/1706.03762. The five-turn Wordle state is short
enough that the dense trunk currently captures the useful interactions more cheaply.

Deep Sets is also not a direct fit for the main state encoder because Wordle turns are chronological, not permutation
invariant: https://arxiv.org/abs/1703.06114. Set-like aggregation may be useful later for derived constraint features,
but it should not replace the ordered turn representation without stronger evidence.

## Experiment Method

All comparable mini experiments used:

```sh
make mini-train TRAIN_ARGS='--seed <seed>'
```

Common settings:

- dataset: `data/mini`
- seeds: `20260613`, `20260614`, `20260615`, `20260616`, `20260617`
- epochs: `300`
- batch size: `32`
- learning rate: `0.01`
- learning-rate decay: enabled
- train batches: all `50` mini batches per epoch
- validation batches: all `50` mini batches per epoch
- final step: `15000`

Metrics were read from the run manifests and cross-checked against TensorBoard event scalars with:

```sh
go run ./cmd/export-scalars --run <run-id>
```

The primary comparison metric is final validation mean loss. `validation_delta_from_start` is still useful, but initial
validation loss differs by architecture because model initialization differs, so final validation loss is the cleaner
architecture comparison.

The full per-run architecture rerun record is committed at `docs/mini-experiment-rerun-2026-06-14.csv`. The mini direct
action-logit follow-up record is committed at `docs/direct-action-logits-experiment-2026-06-15.csv`. The full-dataset
direct action-logit follow-up record is committed at `docs/full-direct-action-logits-experiment-2026-06-15.csv`.

## Branch Coverage

The mini-era experiment branches were run directly. Two older architecture-only branches did not directly support the
seeded mini command, so their model changes were ported onto `mini-data` and run as comparable mini branches:

| Source | Comparable Mini Branch | Notes |
| --- | --- | --- |
| `github/experiment/deeper-dense-trunk` | `experiment/mini-depth-plus-three` | Same `internal/model/policy.go`; already covered directly. |
| `github/experiment/input-and-trunk` | `experiment/mini-input-and-trunk` | Ported input-encoder and trunk architecture to the mini-data base. |
| local `experiment/deeper-dense-trunk` | `experiment/mini-deeper-dense-trunk-wide` | Ported the wider dense trunk to the mini-data base. |

## Five-Seed Results

| Variant | Runs | Avg Final Train | Avg Final Validation | Final Validation SD | Best Final Validation | Worst Final Validation | Avg Validation Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `experiment/mini-deeper-dense-trunk-wide` | 5 | 7.521707 | 7.812392 | 0.023732 | 7.783216 | 7.845639 | -1.970809 |
| `experiment/mini-depth-plus-three` | 5 | 7.545666 | 7.824779 | 0.044374 | 7.777839 | 7.879076 | -1.602316 |
| `experiment/mini-deep-small-tail` | 5 | 7.596449 | 7.851222 | 0.054611 | 7.790408 | 7.911353 | -1.810522 |
| `experiment/mini-input-and-trunk` | 5 | 7.693311 | 7.919675 | 0.013673 | 7.904375 | 7.941321 | -1.458994 |
| `experiment/mini-depth-plus-one` | 5 | 7.698082 | 7.927721 | 0.046617 | 7.873328 | 7.981191 | -2.090435 |
| `mini-data` | 5 | 7.736951 | 7.956361 | 0.036813 | 7.904601 | 7.997783 | -1.580589 |
| `experiment/mini-transformer-state` | 5 | 7.653742 | 7.959821 | 0.084140 | 7.823600 | 8.040271 | -21.602677 |
| `experiment/mini-smaller-output-tail` | 5 | 7.747992 | 7.962559 | 0.030951 | 7.918195 | 8.001349 | -1.445984 |
| `experiment/mini-flat-state-encoder` | 5 | 7.811808 | 8.001186 | 0.019250 | 7.983587 | 8.032782 | -1.035801 |

The wider dense trunk beats the baseline by `0.143969` average final validation loss and beats the deep-small-tail
variant by `0.038831`. Its lead over `experiment/mini-depth-plus-three` is only `0.012387`, so it should be treated as
the current best mini result rather than as a settled architecture choice.

## Paired Checks

Wider dense trunk minus `experiment/mini-depth-plus-three` by seed:

| Seed | Difference |
| ---: | ---: |
| 20260613 | -0.055788 |
| 20260614 | 0.022708 |
| 20260615 | -0.095860 |
| 20260616 | 0.044728 |
| 20260617 | 0.022276 |

Negative means the wider dense trunk was better. The wider trunk won 2 of 5 paired seeds against
`experiment/mini-depth-plus-three`, but its two wins were large enough to give it the better mean.

Best variant per seed:

| Seed | Best Variant | Final Validation | Runner Up | Runner Up Validation | Margin |
| ---: | --- | ---: | --- | ---: | ---: |
| 20260613 | `experiment/mini-deeper-dense-trunk-wide` | 7.809017 | `experiment/mini-depth-plus-three` | 7.864806 | 0.055788 |
| 20260614 | `experiment/mini-depth-plus-three` | 7.801263 | `experiment/mini-transformer-state` | 7.823600 | 0.022336 |
| 20260615 | `experiment/mini-deeper-dense-trunk-wide` | 7.783216 | `experiment/mini-deep-small-tail` | 7.790408 | 0.007192 |
| 20260616 | `experiment/mini-depth-plus-three` | 7.800911 | `experiment/mini-deeper-dense-trunk-wide` | 7.845639 | 0.044728 |
| 20260617 | `experiment/mini-depth-plus-three` | 7.777839 | `experiment/mini-deeper-dense-trunk-wide` | 7.800115 | 0.022276 |

## Direct Action-Logit Follow-Up

Branch `experiment/mini-direct-action-logits` tested the current wider trunk with a direct learned action head instead
of output embeddings. The direct head keeps the same shared turn encoder and trunk through the final 128-wide hidden
state, then emits one learned logit per action:

```text
shared turn encoder: 145 -> 128 -> 64
dense trunk:         321 -> 256 -> 256 -> 256 -> 128 -> 128 -> 128
direct logits:       128 -> action_count
```

This was worse than the output-embedding head on every paired seed:

| Seed | Embedding Final Train | Embedding Final Validation | Direct Final Train | Direct Final Validation | Direct Minus Embedding Validation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20260613 | 7.521885 | 7.809017 | 8.258083 | 8.339715 | 0.530698 |
| 20260614 | 7.526208 | 7.823971 | 8.342547 | 8.391163 | 0.567192 |
| 20260615 | 7.488623 | 7.783216 | 8.219765 | 8.309016 | 0.525800 |
| 20260616 | 7.564678 | 7.845639 | 8.350034 | 8.394972 | 0.549333 |
| 20260617 | 7.507139 | 7.800115 | 8.300537 | 8.364250 | 0.564135 |

Five-seed averages:

| Variant | Runs | Avg Final Train | Avg Final Validation | Final Validation SD | Best Final Validation | Worst Final Validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wider trunk + output embeddings | 5 | 7.521707 | 7.812392 | 0.023732 | 7.783216 | 7.845639 |
| Wider trunk + direct action logits | 5 | 8.294193 | 8.359823 | 0.036135 | 8.309016 | 8.394972 |

The direct-logit head averaged `0.547431` worse final validation loss and also had much worse final train loss, so this
looks like optimization or sample-efficiency trouble rather than merely overfitting. The output-embedding head should
stay in place.

## Full-Data Direct Action-Logit Follow-Up

The same head comparison was rerun against the full training set for 60 epochs on three paired seeds:

```sh
make full-train TRAIN_ARGS='--epochs 60 --seed <seed>'
```

The output-embedding runs used `experiment/mini-deeper-dense-trunk-wide` at commit `2238293`; the direct-logit runs used
`experiment/mini-direct-action-logits` at commit `707e00b`. All runs reached global step `86580`, and the manifest
metrics were cross-checked against scalar export values.

| Seed | Embedding Final Train | Embedding Final Validation | Direct Final Train | Direct Final Validation | Direct Minus Embedding Validation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20260613 | 7.957239 | 8.129922 | 8.446190 | 8.453453 | 0.323530 |
| 20260614 | 7.951001 | 8.153450 | 8.442034 | 8.459470 | 0.306020 |
| 20260615 | 7.992123 | 8.175472 | 8.436596 | 8.460229 | 0.284757 |

Three-seed averages:

| Variant | Runs | Avg Final Train | Avg Final Validation | Final Validation SD | Best Final Validation | Worst Final Validation | Avg Validation Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wider trunk + output embeddings | 3 | 7.966788 | 8.152948 | 0.022779 | 8.129922 | 8.175472 | -1.461844 |
| Wider trunk + direct action logits | 3 | 8.441606 | 8.457717 | 0.003713 | 8.453453 | 8.460229 | -0.086189 |

On the full dataset, the direct-logit head averaged `0.304769` worse final validation loss and lost all three paired
seeds. The output-embedding head remains the better candidate for larger-data work.

## Next Validation Gate

Run the larger-data validation with both top dense-trunk candidates:

```sh
git checkout experiment/mini-deeper-dense-trunk-wide
make full-train TRAIN_ARGS='--seed 20260613'
go run ./cmd/export-scalars --run <run-id>

git checkout experiment/mini-depth-plus-three
make full-train TRAIN_ARGS='--seed 20260613'
go run ./cmd/export-scalars --run <run-id>
```

If compute allows, use the same five seeds for the larger-data comparison. If not, start with seeds `20260613` and
`20260614`; if the result remains close, continue the remaining seeds before choosing between the two dense trunks.
