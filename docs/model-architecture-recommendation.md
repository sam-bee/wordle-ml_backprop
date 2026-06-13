# Model Architecture Recommendation

Date: 2026-06-13

## Recommendation

Adopt the deeper dense-trunk architecture from branch `experiment/mini-depth-plus-three` for the next full-data
validation step:

```text
shared turn encoder: 145 -> 128 -> 64
dense trunk:         321 -> 256 -> 256 -> 128 -> 128 -> 128 -> 64
output embedding:   26 fixed letter-count features + 38 trainable tail features
```

Do not adopt the smaller output tail, flat raw-state encoder, or transformer state encoder yet. The smaller output tail
and transformer improved over the seeded baseline on at least one mini run, but neither matched the deeper trunk. The
smaller tail also hurt when combined with the deeper trunk.

## Why This Direction

The current model already uses ReLU layers with He-style initialization. He et al. derived this initialization to keep
deep rectifier networks trainable from scratch, which makes a moderate-depth MLP trunk a reasonable first capacity
increase for this codebase: https://arxiv.org/abs/1502.01852

Depth should still be treated cautiously. ResNet showed that substantially deeper plain networks can become harder to
optimize and that residual parameterization helps depth scale: https://arxiv.org/abs/1512.03385. The mini experiments
below support adding several trunk layers now, but if we go materially deeper than this recommendation, the next
architecture change should add residual blocks rather than simply stacking more dense layers.

The flat-state encoder experiment tested early cross-turn mixing by removing independent per-turn compression. It did
not win. For this small, fixed-length chronological input, the existing shared turn encoder plus ordered concatenation
appears to be a better inductive bias than a single raw flat state vector. Attention remains a plausible future
direction for richer sequence mixing, but the Transformer motivation is strongest when each position needs to attend to
other positions through learned interactions: https://arxiv.org/abs/1706.03762. The five-turn Wordle state is short
enough that the deeper trunk currently captures the useful interactions more cheaply. A follow-up transformer state
encoder test used one learned state token, five turn tokens, one self-attention block, and the existing action embedding
head; it still trailed the deeper dense trunk on final validation loss.

Deep Sets is also not a direct fit for the main state encoder because Wordle turns are chronological, not permutation
invariant: https://arxiv.org/abs/1703.06114. Set-like aggregation may be useful later for derived constraint features,
but it should not replace the ordered turn representation without stronger evidence.

## Experiment Method

All mini experiments below used:

```sh
make mini-train TRAIN_ARGS='--seed <seed>'
```

Common settings:

- dataset: `data/mini`
- epochs: `300`
- batch size: `32`
- learning rate: `0.01`
- learning-rate decay: enabled
- train batches: all `50` mini batches per epoch
- validation batches: all `50` mini batches per epoch
- final step: `15000`

Metrics were read from TensorBoard event files with:

```sh
go run ./cmd/export-scalars --run <run-id>
```

The primary comparison metric is final validation mean loss. `validation_delta_from_start` is still useful, but initial
validation loss differs by architecture because model initialization differs, so final validation loss is the cleaner
architecture comparison.

## Results

Single-seed sweep, seed `20260613`:

| Variant | Branch | Commit | Run | Initial Validation | Final Train | Final Validation | Validation Delta |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Baseline | `mini-data` | `007499e` | `run-20260613-172528.041117237` | 9.322984 | 7.792625 | 7.997702 | -1.325282 |
| +1 trunk layer | `experiment/mini-depth-plus-one` | `a4a1a85` | `run-20260613-172740.342315233` | 9.106889 | 7.782953 | 7.980847 | -1.126042 |
| +3 trunk layers | `experiment/mini-depth-plus-three` | `c5859d5` | `run-20260613-172952.934200996` | 9.077322 | 7.600320 | 7.864825 | -1.212497 |
| Smaller tail, 22 trainable dims | `experiment/mini-smaller-output-tail` | `cd0e36b` | `run-20260613-173208.501912207` | 9.032999 | 7.770608 | 7.975345 | -1.057654 |
| Flat raw-state encoder | `experiment/mini-flat-state-encoder` | `f1b470d` | `run-20260613-173416.402321445` | 8.938625 | 7.796546 | 7.988999 | -0.949626 |
| +3 trunk layers + smaller tail | `experiment/mini-deep-small-tail` | `3201342` | `run-20260613-173630.945160937` | 9.175889 | 7.684533 | 7.911773 | -1.264116 |

Second-seed check for the baseline and strongest candidates:

| Variant | Seed | Run | Initial Validation | Final Train | Final Validation | Validation Delta |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Baseline | 20260613 | `run-20260613-172528.041117237` | 9.322984 | 7.792625 | 7.997702 | -1.325282 |
| Baseline | 20260614 | `run-20260613-173831.735291220` | 9.618725 | 7.699415 | 7.934891 | -1.683833 |
| +3 trunk layers | 20260613 | `run-20260613-172952.934200996` | 9.077322 | 7.600320 | 7.864825 | -1.212497 |
| +3 trunk layers | 20260614 | `run-20260613-173951.122126400` | 9.431856 | 7.514718 | 7.800358 | -1.631498 |
| +3 trunk layers + smaller tail | 20260613 | `run-20260613-173630.945160937` | 9.175889 | 7.684533 | 7.911773 | -1.264116 |
| +3 trunk layers + smaller tail | 20260614 | `run-20260613-174126.858588574` | 9.700000 | 7.597196 | 7.850456 | -1.849544 |
| Transformer state encoder | 20260613 | `run-20260613-180352.340111860` | 32.376084 | 7.816905 | 8.040507 | -24.335576 |
| Transformer state encoder | 20260614 | `run-20260613-180518.186031000` | 29.058348 | 7.436457 | 7.824831 | -21.233517 |

Two-seed averages:

| Variant | Avg Final Train | Avg Final Validation | Avg Validation Delta |
| --- | ---: | ---: | ---: |
| Baseline | 7.746020 | 7.966297 | -1.504558 |
| +3 trunk layers | 7.557519 | 7.832591 | -1.421998 |
| +3 trunk layers + smaller tail | 7.640864 | 7.881115 | -1.556830 |
| Transformer state encoder | 7.626681 | 7.932669 | -22.784547 |

The +3 trunk reduces average final validation mean loss by about `0.1337` versus baseline across the two checked seeds.
It also beats the combined +3/smaller-tail model by about `0.0485`, so the 38-dimensional trainable output tail should
stay in place for now. The transformer state encoder beats the baseline average by about `0.0336`, but it is about
`0.1001` worse than the +3 trunk average, so it is not the next architecture candidate.

## Next Validation Gate

Use `experiment/mini-depth-plus-three` as the candidate architecture for the larger dataset. The next run should compare
it against the current baseline with the same fixed seeds and the same exporter workflow:

```sh
make full-train TRAIN_ARGS='--seed 20260613'
go run ./cmd/export-scalars --run <run-id>
```

If the deeper plain trunk stops improving or becomes unstable on larger runs, test a residual dense trunk before adding
more plain layers.
