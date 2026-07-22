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
direction. Plans are `H=10` acceleration rows flattened to R^30, and the CFM velocity field is

```text
v_theta(x_s, c_t, s) in R^30,   trunk 48 -> 64 -> 64 -> phi (64) -> 30
```

(48 = 30 plan + 10 context + 8 sinusoidal time features; `phi` is the penultimate representation
the GP uncertainty operates on; controls clamp to the 1 m/s^2 demonstration cap; sampling uses 16
Euler steps.) Implementation: [`flow_model.py`](../safe_mppi/flow_model.py),
[`ball_flow_task.py`](../safe_mppi/ball_flow_task.py).

## 2. Recipe (as specified)

| variable | value |
|---|---:|
| demonstrations | 10 per gamma (`configs/ball_biased_demo.json`, below-biased SafeMPPI) |
| horizon `H` | 10 |
| flow trunk | 64 -> 64, no encoder |
| `K` generated plans | 16 |
| `B` verifier queries | 4 |
| parallel episodes | 2 per gamma |
| RBF buffer cap | 256 |
| replay window `W` | 2 rounds |
| batch size | 32 |
| gradient steps | one exact pass over eligible positives (`inner_steps=None`) |
| learning rate | 1e-5 (3e-5 collapsed the rare right-mode; see section 5) |
| beta | calibrated once with `calibrate_fixed_beta` (ESS target 0.5), then fixed |
| negative loss | alpha = 0 |
| execution | verifier-positive plan minimizing the native (untilted) SafeMPPI cost |

Pretraining: sliding H-step windows over the 40 demo rollouts (1457 windows), 2 extra
geometry-consistent context-jitter copies (0.02 m/m s^-1; the 10-D context is rebuilt exactly from
the perturbed state), 1200 epochs Adam 3e-4 cosine. Reproduce with:

```bash
python scripts/pretrain_ball_flow.py                # demos + CFM + lengthscale/beta calibration
python scripts/run_ball_expansion.py --rounds 60    # B1 expansion + event log
python scripts/evaluate_ball_expansion.py           # untilted raw eval + figures + video
python -m safe_mppi.ball_flow_diagnostics --expansion outputs/ball_flow/expansion
```

## 3. The GREEN verifier and the execution rule

`BallFlowTask.verify` labels each candidate plan with the full-horizon GREEN verifier:

1. dense taskspace containment and strictly positive obstacle clearance;
2. the **rebuilt polytope chain**: at every plan knot the nominal polytope is rebuilt at `q_h`
   (so `H_P(q_h) = 1`) and the next knot must satisfy `H_P(q_{h+1}) >= 1 - gamma`.

The chain is exactly the certificate every executed demonstration step satisfied online (the
logged one-step slack), applied along the whole candidate plan. The rotating tangent face
certifies skirting motion — only the velocity component *toward* an obstacle consumes contraction
budget — whereas a single start-anchored face can never certify a plan that passes the ball
inside the horizon (the wrap blindspot). `hp_eligible` is the separate one-step nominal (BLUE)
gate, and among plans passing both, the executed plan minimizes the native SafeMPPI cost
(running/control/smoothness/terminal — **no z bias**: expansion is untilted).

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

## 5. Results (60-round canonical run; 40-round no-forced-above ablation in parentheses)

Pretraining: 1457 windows, CFM valid loss 1.44 -> 0.70; raw pretrained audit SR 0.34 with modes
{below, left, right} — **above is absent by construction** (the demonstrations are below-biased).
Calibration: RBF lengthscale 0.807, beta 0.0161 (ESS target 0.5).

**Raw temperature-1 closed loop** (canonical start, 16 episodes x 4 gammas per checkpoint):

| checkpoint | SR | CR | route coverage | untilted verifier validity |
|---:|---:|---:|---:|---:|
| round 0 | 0.59 | 0.12 | 0.75 (below/left/right) | 0.36 |
| round 24 | 0.70 | 0.00 | 0.75 | 0.40 |
| round 48 | **0.77** | 0.02 | 0.75 | 0.39 |
| round 60 | 0.64 | 0.05 | 0.75 | 0.36 |

Collision rate collapses to ~0 by round 8 and the three pretrained modes never collapse over 60
rounds (`mode_share.png`) — the lr 3e-5 pilot lost the rare right-mode, which is why the recipe
pins 1e-5. Success improves 0.59 -> ~0.75 with late-round drift; validity rises 0.36 -> 0.42
(round ~24) then relaxes. Gamma trends (`gamma_trend_vs_safemppi.png`): the raw policy is
*uniformly more conservative* than its demonstrator — clearance 0.21-0.23 m at every gamma
(demonstrator: monotone 0.24 -> 0.10) and ~1.2 s slower; the scalar gamma input modulates the
flow far more weakly than it modulates SafeMPPI.

**The support finding (mode discovery).** With fixed canonical starts, zero above-mode plans
appear among ~24k near-ball candidate draws in 20+40 rounds — sigma-tilted acquisition can only
re-rank what the policy samples, and the below-biased prior assigns no mass to climbing over.
With the forced-above replica starts, the GREEN verifier certifies above-corridor positives in
every replay window (44-103 per 15 rounds) and **tilted acquisition covers all four modes from
round 1** (`mode_timeline.png`), yet raw sampling at the canonical start still shows only
below/left/right at round 60; the above-start raw probe improves SR 0.00 -> 0.17 without
producing raw above crossings. Newly acquired modes therefore appear in tilted acquisition but
do not transfer into raw sampling at distant contexts within 60 one-pass rounds: the bottleneck
is prior sample support and context-conditional gating (b_near - p flips direction between the
corridors), not the acquisition rule and not the verifier.

**Decisive acquisition metric** (fixed probe state, 400 draws per round with missing modes):

| regime | P(new valid mode / high-sigma B) | P(new valid mode / uniform B) |
|---|---:|---:|
| canonical 60-round run | **0.0110** | 0.0095 |
| no-forced-above ablation (11 missing-mode rounds) | **0.0060** | 0.0052 |

The inequality holds in both regimes (~+16% relative), with small absolute rates precisely
because valid new-mode candidates are rare in the policy's own K=16 pool — support, not
acquisition, binds.

**Representation probes** (fixed 600-item bank, s=0.9, 4 averaged noise draws; full tables in
[`assets/ball_flow/REPRESENTATION_REPORT.md`](assets/ball_flow/REPRESENTATION_REPORT.md)):
kNN validity 0.94, kNN route-mode 0.75-0.76, linear/RBF validity AUROC 0.96/0.88-0.90,
route-mode probe 0.84-0.85, corr(sigma, nearest-queried-feature distance) 0.53-0.80, and the
control-magnitude shortcut check (0.73) stays below the mode probes — phi_s separates routes and
validity rather than just control magnitude. Flow-time ablation: s=0.5 is clearly worst
(kNN mode 0.65), s=0.8-0.9 best, s=0.95 slightly degraded — more denoised is not monotonically
better, matching the source paper's feature-timestep ablation.

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
- `green_verifier_chain.png` — the GREEN rebuilt-polytope chain along one executed plan with its
  per-knot margins.
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
