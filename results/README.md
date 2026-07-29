# Canonical result snapshots

This directory stores compact, reviewable copies of completed experiment artifacts. Raw
checkpoints and large query archives remain in their authenticated server output roots.

| snapshot | source | interpretation |
|---|---|---|
| [`ball_flow_1815db8`](ball_flow_1815db8/) | PR #2, commit `1815db8` | 3-D ball-flow revision with 40 fan demonstrations per gamma, shallow flow trunk, B1 expansion, raw evaluation, and representation audit |
| [`global50_reference/pretrain_global10_h48p32_s0`](global50_reference/pretrain_global10_h48p32_s0/) | local canonical pretraining contract | 200 successful SafeMPPI demonstrations, portable manifests/calibration features, and the `41 -> 48 -> 32 -> 30` pretrained flow checkpoint used by the deployment bridge |
| [`lab_ball_expansion/minhyuk_nozB5_r20`](lab_ball_expansion/minhyuk_nozB5_r20/) | no-\(z\) sweep, selected round 20 | compact visual-policy r0/r20 endpoint pair and disjoint raw \(M=50/\gamma\) audit; experimental above-route discovery with severe forgetting, not a qualified controller |

Generated expansion/evaluation runs remain ignored. Only the canonical pretraining inputs and model
needed to load or reproduce the deployment bridge are tracked under `global50_reference/`.
