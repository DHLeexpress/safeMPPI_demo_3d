# Noised-representation audit (ball task)

Fixed bank of 600 candidates at the approach state; s=0.9, 4 shared base-noise draws averaged per plan.

## Decisive acquisition metric

- P(new valid route mode | high-sigma B) = **0.0**
- P(new valid route mode | uniform B) = **0.0**
- verdict: **FAIL** — sigma-tilted acquisition does NOT preferentially reach unseen valid route modes (averaged over rounds that still had missing modes).

## Per-round probes (s=0.9)

| round | kNN mode | kNN valid | lin AUROC | rbf AUROC | mode probe | silhouette | corr(sigma,novelty) | kNN |u| shortcut |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.78 | 0.94 | 0.96 | 0.92 | 0.90 | 0.18 | nan | 0.76 |
| 8 | 0.78 | 0.94 | 0.96 | 0.92 | 0.90 | 0.18 | 0.51 | 0.76 |
| 16 | 0.77 | 0.94 | 0.96 | 0.92 | 0.90 | 0.18 | 0.67 | 0.76 |
| 24 | 0.78 | 0.94 | 0.96 | 0.92 | 0.90 | 0.18 | 0.66 | 0.76 |
| 32 | 0.78 | 0.94 | 0.96 | 0.92 | 0.91 | 0.18 | 0.59 | 0.76 |
| 40 | 0.77 | 0.94 | 0.96 | 0.92 | 0.91 | 0.18 | 0.69 | 0.76 |
| 48 | 0.77 | 0.94 | 0.96 | 0.92 | 0.91 | 0.18 | 0.79 | 0.76 |
| 56 | 0.77 | 0.94 | 0.96 | 0.92 | 0.90 | 0.18 | 0.63 | 0.76 |
| 64 | 0.77 | 0.94 | 0.96 | 0.92 | 0.90 | 0.18 | 0.69 | 0.76 |
| 72 | 0.77 | 0.94 | 0.96 | 0.92 | 0.90 | 0.18 | 0.69 | 0.76 |
| 80 | 0.77 | 0.94 | 0.96 | 0.92 | 0.90 | 0.18 | 0.63 | 0.77 |

## Flow-time ablation (final round)

| s | kNN mode | kNN valid | mode probe | silhouette |
|---:|---:|---:|---:|---:|
| 0.5 | 0.67 | 0.94 | 0.71 | 0.03 |
| 0.8 | 0.75 | 0.94 | 0.87 | 0.11 |
| 0.9 | 0.77 | 0.94 | 0.90 | 0.18 |
| 0.95 | 0.76 | 0.94 | 0.89 | 0.17 |

t-SNE is used for visualization only; the claims above rest on the fixed-bank local probes and the acquisition discovery rate.
