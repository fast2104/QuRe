# FashionIQ: internal runs and published benchmark

## Published FashionIQ validation benchmark

The QuRe paper reports FashionIQ **validation** Recall@10 and Recall@50 in
Table 2. The local evaluation also uses the FashionIQ validation split, so these
two metrics can be compared directly at the reported protocol level. Results
below are transcribed from the paper; `Average` is the paper's category-average
column.

| Method | Dress R@10 | Dress R@50 | Shirt R@10 | Shirt R@50 | Toptee R@10 | Toptee R@50 | Average R@10 | Average R@50 | Paper Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CoSMo | 23.60 | 49.18 | 18.11 | 43.18 | 24.63 | 54.31 | 22.11 | 48.89 | 35.50 |
| MGUR | 23.15 | 48.74 | 18.99 | 43.47 | 25.55 | 52.83 | 22.56 | 48.35 | 35.46 |
| CLIP4CIR | 38.32 | 63.90 | 44.31 | 65.41 | 47.27 | 70.98 | 43.30 | 66.76 | 55.03 |
| Bi-BLIP4CIR | 39.12 | 62.92 | 39.21 | 62.81 | 44.37 | 67.06 | 40.90 | 64.26 | 52.58 |
| CoVR-BLIP | 44.55 | 69.03 | 48.43 | 67.42 | 52.60 | 74.31 | 48.53 | 70.25 | 60.24 |
| SPRC | 45.71 | 70.00 | 51.37 | 72.77 | 55.48 | 77.46 | 50.86 | 73.41 | 62.13 |
| **QuRe (paper)** | **46.80** | **69.81** | **53.53** | **72.87** | **57.47** | **77.77** | **52.60** | **73.48** | **63.04** |

Source: Kwak et al., *QuRe: Query-Relevant Retrieval through Hard Negative
Sampling in Composed Image Retrieval*, Table 2, FashionIQ validation results,
arXiv:2507.12416v1, page 4.

The cited page number above is the printed page number (PDF page 5).

The `Paper Avg.` column is copied as printed. In particular, the paper reports
60.24 for CoVR-BLIP, although the arithmetic mean of its listed average R@10
and R@50 (48.53 and 70.25) is 59.39. The local `Average R@10+R@50` comparison
below is calculated consistently from the two average recall values, not from
that printed CoVR-BLIP field.

## Backbone-controlled comparison in the paper

The paper's Appendix Table 7 gives a narrower comparison with matched
backbones. It reports the following `Avg.` values:

| Backbone | Method | Average R@10 | Average R@50 | Paper Avg. |
|---|---|---:|---:|---:|
| BLIP | Bi-BLIP4CIR | 40.90 | 64.26 | 52.58 |
| BLIP | QuRe | 46.27 | 67.89 | 57.08 |
| BLIP-2 | CoVR-2 | 49.34 | 69.91 | 59.63 |
| BLIP-2 | QuRe | 52.60 | 73.48 | 63.04 |

Source: arXiv:2507.12416v1, Appendix Table 7 (PDF page 12).

## Our three runs

These are the local evaluation JSONs. Recall@5 is included for completeness,
but is **not** compared to the published baselines because Table 2 does not
report R@5.

| Run | Training setting | Dress R@5 / R@10 / R@50 | Shirt R@5 / R@10 / R@50 | Toptee R@5 / R@10 / R@50 | Average R@5 / R@10 / R@50 | Average R@10+R@50 |
|---|---|---:|---:|---:|---:|---:|
| No negative sampling | 30 epochs, batch 8, `use_rerank=false` | 31.53 / 40.90 / 65.15 | 39.21 / 47.45 / 68.16 | 39.01 / 49.31 / 71.60 | 36.58 / 45.89 / 68.30 | 57.09 |
| Original code, partial | 5 completed epochs, batch 32, `use_rerank=true` | 27.66 / 36.59 / 62.22 | 36.36 / 45.34 / 68.20 | 40.18 / 48.85 / 72.46 | 34.74 / 43.59 / 67.63 | 55.61 |
| Original code, full | 30 epochs, batch 8, `use_rerank=true` | 36.39 / 46.01 / 69.16 | 42.98 / 52.06 / 71.93 | 45.79 / 56.20 / 77.15 | 41.72 / 51.42 / 72.75 | 62.09 |
| **QuRe (paper)** | Published benchmark, validation | — / 46.80 / 69.81 | — / 53.53 / 72.87 | — / 57.47 / 77.77 | — / 52.60 / 73.48 | **63.04** |

Local result sources:

- No negative sampling: `cir_eval/fiq/new_qformer_2026-10-09_0.json`;
  run config: `experiments/new_qformer_2026-10-09_0/config.json`.
- 5 completed epochs: `cir_eval/fiq/debug_2026-10-08_0.json`;
  run config and epoch log: `experiments/debug_2026-10-08_0/config.json` and
  `experiments/debug_2026-10-08_0/log.txt`.
- 30 epochs, batch 8: `cir_eval/fiq/debug_2026-10-10_4.json`;
  run config: `experiments/debug_2026-10-10_4/config.json`.

## Interpretation

- The 30-epoch, batch-8 run is the strongest of these local experiments.
- Its average R@10 is 1.18 percentage points below the paper's QuRe result;
  its average R@50 is 0.73 points below. Its mean of average R@10 and R@50 is
  62.09 versus the paper's 63.04.
- It exceeds the published SPRC average R@10 (51.42 vs. 50.86), but is below
  SPRC average R@50 (72.75 vs. 73.41). Thus it does **not** establish a new
  SOTA result under the paper's combined average metric.
- The 5-epoch run is incomplete, so it is useful as a debugging/reference run,
  not as a fair final-model comparison.
- These published values are the SOTA comparison reported by the 2025 QuRe
  paper; this table does not claim to cover later work published after that
  paper.
- The 30-epoch run's saved config records `model=blip2`,
  `backbone=pretrain_vitL`, and `blip_size=base`, while also retaining
  `image_encoder=resnet50_layer4`. The paper describes BLIP-2 with a ViT-L
  image encoder. The result is evaluated on the same validation split and
  metrics, but confirm which config fields determine the instantiated model
  before presenting the run as an exact reproduction of the paper's setup.

The official comparison source is
[arXiv:2507.12416](https://arxiv.org/abs/2507.12416) and its
[PDF](https://arxiv.org/pdf/2507.12416).
