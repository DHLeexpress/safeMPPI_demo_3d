# Ball-below task: z-biased SafeMPPI grazing under a 20-inch ball

## Task definition

| item | value |
|---|---|
| start | position `(0, 0, 2)` m, velocity `0` |
| goal | `(3, 0, 2)` m, reach radius `0.15` m |
| obstacle | one sphere ("20-inch ball", radius `0.254` m) at the segment midpoint `(1.5, 0, 2)` |
| taskspace | `[-1,4] x [-2,2] x [0,4]` m (start/goal strictly interior) |
| requirement | every rollout must pass **below the ball's latitude-0 circle** (its `z=2` equator), forming a smooth, roughly symmetric grazing valley |
| demonstration control cap | `1.0 m/s^2` per axis (intentionally reduced from 1.2) |
| warm start | first plan is seeded with `[0.1, 0, 0] m/s^2` so the robot immediately sets off toward the goal |
| gamma grid | `0.1, 0.3, 0.5, 1.0`, ten paired seeds each |

Everything lives in [`ball_below_config.json`](../ball_below_config.json); run with

```bash
python run.py --config ball_below_config.json --output outputs/ball_below_final --device cpu
python -m safe_mppi.ball_analysis --run outputs/ball_below_final
```

## What was added to the package

1. **Exponential altitude bias** (`controller.py`): every rollout step pays
   `z_bias_weight * exp((z - z_bias_ref) / z_bias_temperature)`, clamped at `exp(60)`.
   With `ref=2`, `w=0.4`, `T=0.025` the term is a one-sided wall: at `z=1.9` it is
   `0.4*exp(-4) ~ 0.007` (negligible, as requested), at `z=2` it is `0.4`, at `z=2.1` it is `22`.
   Rollouts are therefore biased below the ball's latitude 0 and cannot sneak over the top.
2. **Warm-start bias** (`controller.py`): when no previous plan exists, the nominal control
   sequence is `warm_start_bias` instead of zeros.
3. Both knobs are optional config fields with inactive defaults (`config.py`); the default
   experiment is unchanged and the original tests still pass.
4. **`safe_mppi/ball_analysis.py`**: per-gamma seed overlays (3D + BLUE nominal polytope with its
   level sets near the ball + x-z side view against the latitude-0 line), a cross-gamma overlay,
   gamma-trend panels, and `ball_metrics.json/csv` with symmetry / below-latitude / smoothness /
   saturation numbers per run.

## Final cost recipe (only state, control, terminal + the z penalty)

Soft proximity preference and the progress reward are **disabled** (`0.0`). The surviving costs:

| term | weight | note |
|---|---:|---|
| running goal `\|p-g\|^2` | 2.0 | isotropic in y/z, so the detour direction is chosen only by the z bias |
| control `\|u\|^2` | 1.0 | |
| control smoothness `\|du\|^2` | 1.0 | increased from the 0.12 default |
| terminal goal `\|p_H-g\|^2` | 20.0 | `~ 2` per horizon step, i.e. the state/terminal channels stay balanced |
| z bias `0.4 exp((z-2)/0.025)` | — | the only symmetry breaker |

Sampling: `1024` samples, `sigma=(0.35, 0.20, 0.35)`, MPPI temperature `0.3`,
centroid mixture calmed to `gain=0.05`, `sigma_aniso=1.5`.

The pure equal-weight setting (`1/1/1`, MPPI temperature 0.1, sigma 1.0) was measured first: it is
safe and below-latitude but does not converge (7/20 timeouts) because the near-goal cost landscape
is flatter than the sampling noise, and the strong `w_z=1` wall parks episodes below the reach
ball. The final recipe keeps the state:terminal per-step channels equal and applies the small
position-control bias the task brief anticipated.

## Measured gamma trends (10 seeds per gamma, all 40/40 successful, all below latitude 0)

| gamma | avg min clearance [m] | avg time to goal [s] | mean \|du\| [m/s^2] | \|u\|>0.95cap fraction |
|---:|---:|---:|---:|---:|
| 0.1 | **0.222** | **6.82** | 0.419 | **0.034** |
| 0.3 | 0.104 | 6.39 | 0.372 | 0.054 |
| 0.5 | 0.134 | 6.36 | 0.375 | 0.054 |
| 1.0 | 0.145 | 6.50 | 0.369 | **0.062** |

- **Clearance**: smallest gamma is clearly the most conservative (0.22 m, twice the others); the
  relation is *not* monotone through the middle — gamma 0.3 grazes closest. With a single tangent
  face per obstacle, gamma 0.1 treats a constant-clearance under-pass as "approaching the
  boundary", so it swings wide instead of hugging.
- **Time to goal**: gamma 0.1 is the slowest (6.82 s); gamma 1.0 ticks back up versus 0.3/0.5
  because late braking costs a small overshoot loop at the goal.
- **Smoothness / bang-bang**: cap-saturation rises monotonically with gamma — gamma 1.0 is the
  most bang-bang, as theory predicts. Moderate gammas (0.3/0.5) have the lowest control jitter;
  gamma 0.1 is the jitteriest because the tight contraction keeps fighting the sampler.
- **Symmetry**: mean |z(1.5+d)-z(1.5-d)| is 0.17-0.22 m. The valley bottom lands ~0.2-0.3 m past
  the ball: with a 1 s lookahead the climb-out can only start once the tangent face releases, so a
  perfectly centered dip is out of reach for this horizon; the family is otherwise a smooth,
  tight, symmetric-looking graze.

## Figures

Checked-in copies of the final run live in [`docs/assets/ball_below/`](assets/ball_below/);
`run.py` + `ball_analysis` regenerate them in the output directory.

All gammas, all forty rollouts (3D + x-z side view against the latitude-0 line):

![All gammas overlay](assets/ball_below/ball_all_gammas.png)

Per-gamma 10-seed overlays with the BLUE nominal polytope and its ten `H_P` level sets at the
step nearest the ball:

![gamma 0.1](assets/ball_below/ball_gamma_0.1.png)
![gamma 0.3](assets/ball_below/ball_gamma_0.3.png)
![gamma 0.5](assets/ball_below/ball_gamma_0.5.png)
![gamma 1.0](assets/ball_below/ball_gamma_1.png)

Measured gamma trends (dots are single seeds, line is the mean):

![gamma trends](assets/ball_below/ball_gamma_trends.png)

## Tuning journey (what mattered)

| change | effect |
|---|---|
| `w_z` 1.0 -> 0.4 with sharp `T=0.025` | stops the start-line dive; wall above `z~2.05` still absolute |
| reach radius 0.25 -> 0.15 | removes the "finish low inside the reach ball" shortcut |
| terminal 1 -> 20 | fixes near-goal wandering (7/20 -> 0/20 timeouts) |
| running 1 -> 2 | arrests the post-ball sink, brakes the goal overshoot |
| centroid `gain 0.2 -> 0.05`, `aniso 2.5 -> 1.5` | the escape mixture was injecting full-cap downward kicks near the ball (bottom 1.1 m -> 1.45 m) |
| samples 512 -> 1024, `sigma_y` 0.35 -> 0.2 | kills the rare gamma-0.1 all-infeasible flee/out-of-bounds episode and the diagonal side-sneaks |
| smooth weight 2.0, `sigma_y` 0.12 (tried) | **reverted** — sluggish committed arcs made gamma 0.1 wander again |
