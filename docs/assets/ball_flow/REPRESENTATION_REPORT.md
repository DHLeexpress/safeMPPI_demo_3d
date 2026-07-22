# Noised-representation audit (ball task)

Fixed bank of 600 candidates at the approach state; s=0.9, 4 shared base-noise draws averaged per plan.

## Decisive acquisition metric

- P(new valid route mode | high-sigma B) = **0.011041666666666667**
- P(new valid route mode | uniform B) = **0.009479166666666667**
- verdict: **PASS** — sigma-tilted acquisition does preferentially reach unseen valid route modes (averaged over rounds that still had missing modes).

## Per-round probes (s=0.9)

| round | kNN mode | kNN valid | lin AUROC | rbf AUROC | mode probe | silhouette | corr(sigma,novelty) | kNN |u| shortcut |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.76 | 0.94 | 0.97 | 0.90 | 0.85 | 0.09 | nan | 0.73 |
| 6 | 0.76 | 0.94 | 0.96 | 0.90 | 0.84 | 0.09 | 0.62 | 0.73 |
| 12 | 0.76 | 0.94 | 0.96 | 0.89 | 0.84 | 0.09 | 0.53 | 0.73 |
| 18 | 0.76 | 0.94 | 0.96 | 0.89 | 0.84 | 0.09 | 0.65 | 0.72 |
| 24 | 0.76 | 0.94 | 0.96 | 0.89 | 0.84 | 0.10 | 0.80 | 0.73 |
| 30 | 0.75 | 0.94 | 0.96 | 0.89 | 0.84 | 0.10 | 0.69 | 0.73 |
| 36 | 0.75 | 0.94 | 0.96 | 0.88 | 0.84 | 0.10 | 0.80 | 0.73 |
| 42 | 0.75 | 0.94 | 0.96 | 0.88 | 0.84 | 0.10 | 0.73 | 0.73 |
| 48 | 0.75 | 0.94 | 0.96 | 0.88 | 0.84 | 0.10 | 0.71 | 0.73 |
| 54 | 0.75 | 0.94 | 0.96 | 0.88 | 0.84 | 0.10 | 0.76 | 0.73 |
| 60 | 0.75 | 0.94 | 0.96 | 0.88 | 0.84 | 0.10 | 0.68 | 0.73 |

## Flow-time ablation (final round)

| s | kNN mode | kNN valid | mode probe | silhouette |
|---:|---:|---:|---:|---:|
| 0.5 | 0.65 | 0.94 | 0.73 | 0.02 |
| 0.8 | 0.74 | 0.94 | 0.86 | 0.05 |
| 0.9 | 0.75 | 0.94 | 0.84 | 0.10 |
| 0.95 | 0.72 | 0.94 | 0.84 | 0.09 |

t-SNE is used for visualization only; the claims above rest on the fixed-bank local probes and the acquisition discovery rate.
