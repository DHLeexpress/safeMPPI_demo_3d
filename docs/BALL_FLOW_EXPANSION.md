# Safe Flow Expansion on the ball task: pretraining, deployment, and representation audit

This document wires the task-neutral B1 loop ([`expansion.py`](../safe_mppi/expansion.py)) to the
20-inch-ball task, end to end: biased demonstrations -> conditional-flow pretraining -> verifier-
gated self-generated expansion -> raw (untilted) evaluation -> noised-representation audit.

## 1. Generative policy structure

The context is the smallest geometrically meaningful one (10-D):

```text
c_t = [ g - p_t,  v_t,  b_near - p_t,  gamma ]  in R^10
```

`b_near` is the closest point on the ball surface, so one vector carries obstacle distance and
direction. Plans are `H=10` acceleration rows flattened to R^30. The portable canonical checkpoint
uses raw scalar flow time and the shallow two-layer trunk:

```text
v_theta(x_s, c_t, s) in R^30,   41 -> 48 -> 32 (= phi_s) -> 30
```

(41 = 30 plan + 10 context + 1 raw-time feature; `phi_s` is the second hidden layer — the
penultimate representation used by RBF uncertainty; controls clamp to the 1 m/s^2 demonstration
cap; sampling uses 16 Euler steps.) The implementation also supports the explicitly paired noised
point `(1-s)x_0+sU`, using the exact base `x_0` that generated each sampled plan. Implementation:
[`flow_model.py`](../safe_mppi/flow_model.py) (`trunk_depth=2`, `time_features=raw1`) and
[`ball_flow_task.py`](../safe_mppi/ball_flow_task.py).

## 2. Portable pretraining contract and expansion controls

| variable | value |
|---|---:|
| demonstrations | 50 successful SafeMPPI trajectories per gamma |
| horizon `H` | 10 |
| flow trunk | `48 -> 32`, trunk depth 2, no encoder |
| pretraining windows / epochs | 7,039 / 500 |
| calibrated RBF length scale | 0.298804 |
| calibrated initial beta | 0.102375 |
| raw pretraining audit | SR 0.4625, window validity 0.9040 |

The expansion executable exposes `K`, `B`, parallel episodes, replay window, GP cap, fixed or
adaptive beta, exact-pass or bounded optimizer updates, signed negative gradients, current/frozen
representation references, and query-level or successful-trajectory archives. These are
experimental controls rather than one hidden canonical recipe. Reproduce the checked-in
pretraining bundle or start a new expansion with:

```bash
python scripts/pretrain_ball_flow.py
python scripts/run_ball_expansion.py \
  --pretrain-dir results/global50_reference/pretrain_global10_h48p32_s0 \
  --output outputs/ball_flow/expansion
python scripts/evaluate_ball_expansion.py \
  --pretrain-dir results/global50_reference/pretrain_global10_h48p32_s0 \
  --expansion outputs/ball_flow/expansion
python -m safe_mppi.ball_flow_diagnostics --expansion outputs/ball_flow/expansion
python scripts/render_coverage_video.py --expansion outputs/ball_flow/expansion --stride 5
```

## 3. The GREEN verifier and the execution rule

`BallFlowTask.verify` labels each candidate plan with the full-horizon GREEN verifier:

1. executed-first-segment taskspace/corridor containment and strictly positive
   full-H obstacle clearance; the unexecuted tail is not taskspace-gated;
2. the cloned verifier's **trajectory-fitted variable faces**, generalized from 2-D disks to 3-D
   spheres without changing their constraints. For a window centered at `c`, it fits a unit normal
   `a` and maximum margin `m=a^T(o-c)-r` such that
   `a^T(q_h-c) <= [1-(1-gamma)^h] m` for every horizon step. The real-obstacle faces are augmented
   by 80 fitted artificial-sphere faces in the nominal icosphere directions, bounding GREEN at the
   effective sensing radius exactly as the cloned package bounds its 2-D verifier.

The BLUE online nominal polytope and GREEN post-hoc verifier are therefore distinct objects.
The BLUE one-step value is retained only for diagnosis; it is not an execution gate. A candidate
is execution-eligible only if `q[h+1,x]-q[h,x] > 0` for every plan knot; endpoint distance-to-goal
is not used. Among GREEN-positive, forward-monotone candidates, the executed plan minimizes the
native SafeMPPI cost
(running/control/smoothness/terminal — **no z bias**).

Episode replicas: the even replica of every gamma always starts at the canonical start; with
start diversity the odd replica starts from a randomized collision-free pre-ball state (half of
them above/ahead of the ball, where any forward crossing is geometrically an *above* route).
This is how the task preserves route support — plans are still policy-sampled and verifier-gated.

## 4. Evaluation protocol

Raw evaluation is deliberately bare: closed-loop temperature-1 rollouts from the canonical start
(one sampled plan per step, execute the first action; no verifier, no tilt, fresh seed bank), per
checkpoint and gamma. The validity probe samples raw open-loop plans at three fixed states and
scores them with the GREEN verifier. Route modes are the angular quadrant at the ball-plane
crossing (+y = left, viewed from the start).

## 5. Results (canonical: fan demonstrations, 40/gamma, shallow trunk, beta 0.003, 80 rounds)

**Demonstrations** now span the z<2 side of the ball with a wide angular fan
([`assets/ball_flow/fan_demonstrations.png`](assets/ball_flow/fan_demonstrations.png),
`configs/ball_fan_demo.json`): crossing angles 5-160 degrees w.r.t. the -y axis, strictly below
the equator near the ball (audited max z 1.973). Pretraining on all 160 demos (5672 windows):
raw SR 0.66, modes {below, left, right} — `above` is the single novel mode.

**Automated (data x model) discovery sweep** (`scripts/sweep_ball_flow.py`,
[`assets/ball_flow/sweep_grid.png`](assets/ball_flow/sweep_grid.png)): 300-epoch pretrains,
20-round canonical-start expansions, beta 0.003. It exposes an exploration-competence tradeoff:
10 demos/gamma keeps broad tails (up to 71 above candidates, 11 verified above positives per run)
but SR only ~0.4; 40 demos/gamma trains a strong prior (raw SR -> 0.75-0.77 through expansion)
whose tails collapse onto the demo fan (0-3 above candidates). Depth 2 vs 3 is not decisive.
Reliable discovery therefore needs the strong prior *plus* replica start diversity, which is the
canonical configuration below.

**Canonical run** (40/gamma, trunk_depth 2, beta 0.003 near-greedy acquisition, replica start
diversity, 80 rounds):

| checkpoint | SR | CR | route coverage | untilted validity |
|---:|---:|---:|---:|---:|
| round 0 | 0.75 | 0.08 | 0.75 | 0.35 |
| round 30 | **0.95** | **0.00** | 0.75 | 0.38 |
| round 80 | 0.91 | 0.00 | 0.50-0.75* | 0.36 |

*right-mode raw mass thins late under the novelty-skewed replay (present in the independent
video seed bank at rounds 60-80, absent in the eval bank — a seed-level effect).

Above-mode discovery is now **sustained**: 323 above verifier positives across 80 rounds
(163/56/21/83 per 20-round bucket) — every replay window contains the novel mode, versus exactly
one above positive in the original sharp-bias/deep-model configuration. Raw sampling at the
canonical start still shows below/left/right; the above-start raw probe reaches SR 0.375. The
remaining gap is context-conditional transfer, not acquisition.

**Decisive acquisition metric across regimes**: with weak priors the inequality holds
(10-demo cells and the earlier deep runs: 0.0110 vs 0.0095 and 0.0060 vs 0.0052, ~+16%
relative); with the strong 40-demo prior it is vacuous at the canonical probe state (0 vs 0 —
the policy proposes no above candidates there at all), which is the support-limitation statement
in its sharpest form. The automated report
([`assets/ball_flow/REPRESENTATION_REPORT.md`](assets/ball_flow/REPRESENTATION_REPORT.md))
prints the canonical-run verdict verbatim.

**Final coverage video** ([`assets/ball_flow/coverage_video.mp4`](assets/ball_flow/coverage_video.mp4)):
expansion iterations progressively cover the ball — solid raw trajectories (below/left/right fan)
plus dotted self-generated verifier-positive episodes wrapping the top — ending on the achieved
metrics (SR 0.90, CR 0.00, validity 0.39 on the video seed bank).

**Representation probes** (fixed 600-item two-state bank, s=0.9, 4 averaged noise draws; full
tables in [`assets/ball_flow/REPRESENTATION_REPORT.md`](assets/ball_flow/REPRESENTATION_REPORT.md)):
kNN validity 0.94, kNN route-mode 0.77-0.78, linear/RBF validity AUROC 0.96/0.92, route-mode
probe 0.90, mode silhouette 0.18, corr(sigma, nearest-queried-feature distance) 0.51-0.67, and
the control-magnitude shortcut check (0.76) stays below the mode probe — phi_s separates routes
and validity rather than control magnitude. Flow-time ablation: s=0.5 worst (kNN mode 0.67),
s=0.9 best (0.77), s=0.95 slightly degraded — more denoised is not monotonically better,
matching the source paper's feature-timestep ablation.

## 6. Figures

All curated copies live in [`docs/assets/ball_flow/`](assets/ball_flow/):

- `expansion_curves.png` — closed-loop outcomes, selected-query validity, ESS/K, positive loss
  (straight from `metrics.jsonl` via `expansion_visualize.plot_expansion_results`).
- `raw_curves.png` — raw SR / CR / route coverage / untilted verifier validity across rounds.
- `mode_share.png`, `raw_crossing_fan.png`, `raw_gallery.png` — what the raw policy actually
  flies, colored by route mode.
- `mode_timeline.png` — when each mode first appears in tilted acquisition vs raw sampling.
- `gamma_trend_vs_safemppi.png` — clearance / time-to-goal vs gamma against the demonstrator.
- `sigma_tilt_anatomy.png` — one acquisition step: K plans colored by sigma, the acquisition
  softmax, the selected B, and their verifier verdicts.
- `sigma_mode_decay.png` — per-mode mean sigma of near-ball candidates across rounds (novelty
  decays as modes are queried).
- `legacy_rebuilt_nominal_chain.png` — rebuilt-nominal-chain diagnostic; it must not be interpreted
  as the current GREEN fitted-face safety label.
- `mechanism.mp4` — the `render_expansion_mechanism` video (K plans, selected-B uncertainty,
  verifier positives, executed action, accumulated positive/NVP states).
- `tsne_panels.png`, `tsne_by_flow_time.png`, `representation_probes.png`,
  `cluster_vs_coverage.png` — the representation audit (section 7).

## 7. Noised-representation audit (automated)

`python -m safe_mppi.ball_flow_diagnostics --expansion <dir>` runs the fixed-bank protocol: one
frozen 600-candidate bank (pretrained samples + handcrafted above/below/left/right arcs + uniform
random plans at a fixed approach state, all four gammas), identical base-noise draws (4, averaged)
and flow times across every checkpoint, GP state rebuilt per round from the query archive. It
emits `REPRESENTATION_REPORT.md` + `diagnostics.json` with:

- t-SNE panels on one shared embedding (color = validity / route mode / clearance / GP sigma;
  marker = gamma; opacity = round) — visualization only;
- kNN validity & route-mode accuracy, linear + RBF validity AUROC, route-mode probe accuracy,
  and a control-magnitude shortcut check;
- `|phi_i - phi_j|` vs trajectory distance and sigma vs nearest-queried-feature distance;
- the decisive acquisition metric
  `P(new valid route mode | high-sigma B) > P(new valid route mode | uniform B)`;
- the flow-time ablation `s in {0.5, 0.8, 0.9, 0.95}`.

Claims about the representation rest on those probes, not on t-SNE separation.
