# PRE2-based CFM–MPPI · 3-D bowling contract

This is an isolated inference-time baseline. It does not modify PRE2, S4,
SafeMPPI, or the currently published site.

## Controller

- Generative prior: immutable PRE2 checkpoint, NFE 16, raw H10×3 acceleration
  output.
- Guided proposals: 32 per receding-horizon step.
- Refinement: native-cost top 8, 32 Gaussian perturbations per elite
  (`sigma=0.20`), per-elite softmax mean (`lambda=0.10`), then native-cost
  minimum among the 8 refined plans.
- Deployment dynamics: the exact stateful 3-D reference governor, including
  raw acceleration clipping, acceleration smoothing, dense substeps, total
  speed cap, and vertical speed cap.
- Selection exclusions: no H_P/NVP verifier, progress eligibility label, or
  post-hoc validity label participates in proposal/refinement selection.

## Refinement cost

All three ranking stages use the bowling task's configured SafeMPPI soft cost:

- running and terminal goal distance,
- raw control magnitude,
- raw command smoothness against the previous raw command,
- progress reward,
- configured soft clearance (zero in this pinned bowling task), and
- task-space exponential penalty.

The native SafeMPPI H_P infeasibility gate is intentionally excluded because
CFM–MPPI is the soft-guidance/soft-refinement baseline, not another SafeMPPI
controller.

## Guidance

The sphere and wall CBF share a signed-clearance definition so all constraints
have compatible units:

`h = signed clearance - 0.10 m`, `residual = h_dot + alpha*h`.

The five worst residuals per horizon knot are weighted 5→1. Goal guidance is
negative terminal distance. Both gradients preserve the reference method's
batch-global normalization and early-action markup.

Displayed coefficients are normalized to `[0,1]`:

- displayed reward 1.0 = raw goal coefficient 0.25,
- displayed safety 1.0 = raw safety coefficient 1.0.

The fixed calibration found that raw safety 0.5 still collided in 2/2 matched
gamma=0.5 trials, raw safety 1.0 succeeded in 2/2 with mean 0.165 m clearance,
and raw safety 2.0 regressed to 1/2. For goal guidance, raw 0.125, 0.25, and
0.5 all produced the expected reward-dominant collision, with larger values
increasing penetration; raw 0.25 is therefore the moderate middle endpoint.
The matched balanced endpoint (raw goal 0.25 + raw safety 1.0) succeeded 2/2,
with validity 1.0 and mean clearance 0.138 m. A bounded follow-up selected
`alpha=0.5` and a half-strength balanced reward (raw goal 0.125): on 16 paired
trials this produced 13 successes, no OOB/timeouts, validity 0.916, and mean
successful clearance 0.077 m. The same alpha with safety-only guidance produced
11/16 successes, 0.871 validity, 0.087 m clearance, and a longer 7.75 s TtG.

## Approval regimes

- safety-dominant: `(reward, safety) = (0, 1)`
- reward-dominant: `(reward, safety) = (1, 0)`
- balanced: `(reward, safety) = (0.5, 1)`

The approval candidates use `alpha=0.5`; the full-strength `(1,1)` balanced
setting is retained as a boundary ablation rather than promoted.

The approval comparison uses the same four rollout seeds for every gamma and
every regime, so the displayed qualitative and quantitative comparison is
paired rather than cherry-picked. Reward-dominant also retains four additional
seeds per gamma; all 32 reward-dominant trials collided.
