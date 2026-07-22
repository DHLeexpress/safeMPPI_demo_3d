# safeMPPI demo in 3D

A small, standalone reference for collecting 3D SafeMPPI rollouts from a point-mass double
integrator. The package exposes the taskspace and controller recipe in one JSON file, runs the full
gamma grid, records every actual rollout, reports safety/performance metrics, and renders the BLUE
nominal polytope with its ten horizon level sets.

![Four-pillar in-distribution trajectories colored by gamma](docs/assets/pillars_id_gamma_overlay.png)

This repository intentionally contains one sampling implementation:
`mode1_centroid_anisotropic`. It does not hide alternate sampling modes behind configuration flags.

## Project status

| stage | status | where |
|---|---|---|
| Define taskspace, start/goal, obstacles, and double-integrator dynamics | implemented | [`default_config.json`](default_config.json), [`environment.py`](safe_mppi/environment.py) |
| Define the current mode-1 SafeMPPI recipe | implemented | [`controller.py`](safe_mppi/controller.py) |
| Acquire data for every gamma, compute metrics, and render figures | implemented | [`acquire.py`](safe_mppi/acquire.py), [`visualize.py`](safe_mppi/visualize.py) |
| Safe flow expansion | placeholder only | [`expansion.py`](safe_mppi/expansion.py) |

The expansion entry point raises `NotImplementedError` deliberately. No acceptance rule, learned
model, or training recipe has been guessed.

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --config default_config.json --output outputs/default_run --device cpu
```

CUDA is optional. To use physical GPU 2 while keeping the process-local device name unambiguous:

```bash
CUDA_VISIBLE_DEVICES=2 python run.py \
  --config default_config.json \
  --output outputs/default_run \
  --device cuda:0
```

The run produces one NPZ file per episode, `manifest.json`, `metrics.csv`, `metrics.json`, a 3D
gamma-colored rollout overlay, and a BLUE nominal-level-set figure.

## Ball-below variant

[`ball_below_config.json`](ball_below_config.json) + [`docs/BALL_BELOW.md`](docs/BALL_BELOW.md)
define a second experiment: start `(0,0,2)` to goal `(3,0,2)` with a 20-inch ball at `(1.5,0,2)`,
an exponential altitude penalty `w exp((z-2)/T)` that biases every rollout below the ball's
latitude-0 circle, a `1 m/s^2` demonstration cap, and a `[0.1,0,0]` warm-start bias. Analyze a
finished run with `python -m safe_mppi.ball_analysis --run <output_dir>`.

## Default experiment

All user-facing values live in [`default_config.json`](default_config.json).

| setting | default |
|---|---:|
| taskspace | `[0,5] x [0,5] x [0,3]` m |
| start | position `(0,0,2)` m, velocity `(0,0,0)` m/s |
| goal | `(5,5,2)` m |
| dynamics | 3D double integrator, `dt=0.1 s` |
| planning horizon | `H=10` |
| samples per control period | `512` |
| gamma grid | `.1,.2,.3,.4,.5,.7,1` |
| Gaussian sigma | `(1,1,1) m/s^2` |
| MPPI temperature | `0.1` |
| demonstration control cap | `1.2 m/s^2` per axis |
| platform authority, metadata only | `3.0 m/s^2` per axis |
| sensing range | `2.0 m` |
| nominal base | 80 triangular faces |
| smoothness weight | `0.12` |
| soft proximity preference | `25 relu(0.10-clearance)^2` |
| point-robot/hard obstacle/safety/gain/ZOH margins | all exactly `0` |
| paired episodes | one seed per gamma |

The default scene has no obstacles. It is a configuration and execution smoke test, not a safety
benchmark. Consequently, collision rate is zero and clearance is reported as `null`.

## Dynamics and task geometry

The state is `x=[p,v]` in R6 and the controller commands acceleration `u` in R3:

```text
p[k+1] = p[k] + dt v[k] + 0.5 dt^2 u[k]
v[k+1] = v[k] + dt u[k]
```

The taskspace is a rectangular box. Static sphere rows are `[x,y,z,radius]`; vertical cylinder rows
are `[x,y,radius]` and span the full taskspace height:

```json
"obstacles": {
  "spheres": [[2.5, 2.5, 2.0, 0.4]],
  "cylinders": [[3.5, 1.5, 0.3]]
}
```

The requested default start and goal lie on taskspace faces. They are not silently shifted inward.
That makes the corresponding box margin zero at those points; visualization therefore chooses a
strict-interior rollout state when it needs explicit polytope vertices.

## BLUE nominal polytope and `H_P`

At every control period, [`geometry.py`](safe_mppi/geometry.py) builds

```text
P_k = 2 m sensing polyhedron
      intersect obstacle tangent halfspaces
      intersect taskspace box halfspaces.
```

The sensing polyhedron comes from a once-subdivided icosahedron: 42 sphere vertices define 80
nearly uniform triangular faces. There is no Fibonacci direction index. Sphere and cylinder faces
are tangent to the raw obstacle radius because every hard margin is fixed to zero.

For `P={x: A x <= b}` centered at the current robot position `c`, the normalized field is

```text
H_P(x) = min_i (b_i - a_i^T x) / (b_i - a_i^T c).
```

`H_P(c)=1` for a strict-interior center and `H_P(x)=0` on the active boundary. The geometry audit
below shows the shared 80 angular directions and the nonuniform radial samples used by the broader
3D experiment: 5 cm spacing through 0.2 m, 10 cm spacing through 0.5 m, then 1 m and 2 m context.

![Uniform triangular nominal and H_P construction](docs/assets/uniform_triangular_hp_audit.png)

## The only sampler: mode 1

[`Mode1SafeMPPI`](safe_mppi/controller.py) shifts the previously optimized control sequence, adds
Gaussian perturbations, and biases a subset of samples toward the nominal-polytope centroid when the
local polytope becomes tight. Noise along that centroid direction is anisotropically amplified.

Each sampled horizon is rejected when any step violates

```text
H_P(q[h+1]) >= (1-gamma) H_P(q[h]).
```

Smaller gamma contracts admissible motion more gradually; gamma 1 admits the raw `H_P>=0`
boundary immediately. The cost ranks feasible samples using goal error, control effort, the
user-approved control-smoothness term, progress, and a soft proximity preference. The proximity
term changes ranking only—it does not enlarge obstacles.

Only the first averaged action is executed. The next control period senses again and rebuilds the
polytope. If every sampled horizon is infeasible, the implementation explicitly ranks samples by
their worst `H_P` value; the saved `online_one_step_slack` lets downstream checks detect whether the
executed fallback remained admissible.

## Outputs and metric definitions

For every gamma and seed, `run.py` saves:

- `states`, `controls`, and the exact per-step `poly_A/poly_b` in compressed NPZ form;
- sampled feasible fraction and executed one-step barrier slack;
- success rate (`SR`): reached the goal without collision or taskspace violation;
- collision rate (`CR`): any negative dense-time obstacle clearance;
- average minimum obstacle clearance;
- average time to reach the goal among reaching episodes; and
- taskspace-violation rate and average planning time.

The checked-in one-seed reference metrics are in
[`examples/default_metrics.csv`](examples/default_metrics.csv). These are mechanism checks, not
M25/M100 statistical claims.

| default output | nominal level sets |
|---|---|
| ![Default gamma rollout overlay](docs/assets/default_gamma_rollouts_3d.png) | ![Default BLUE nominal level sets](docs/assets/default_nominal_levelsets_by_gamma.png) |

## ID/OOD mechanism gallery

The repository also preserves the generated conceptual and actual-rollout visuals from the dense
pillar experiment. They demonstrate the uniform `H_P` field, all ten horizon levels, gamma masking,
and the distinction between the BLUE online nominal polytope and GREEN post-hoc SOCP verifier.

| in-distribution: four pillars/open middle | OOD: restored center pillar + overlapping spheres |
|---|---|
| ![ID gamma mask](docs/assets/gamma_mask_id.gif) | ![OOD gamma mask](docs/assets/gamma_mask_ood.gif) |

See the [complete per-gamma visual atlas](docs/VISUALS.md) for all 14 actual rollout GIFs and their
outcomes. The GIFs are evidence imported from the broader experiment; `run.py` in this minimal
repository generates the two PNG figures above, not those detailed three-panel animations.

## Code reading order

1. [`default_config.json`](default_config.json) — the entire public experiment contract.
2. [`safe_mppi/config.py`](safe_mppi/config.py) — typed loading and strict zero-margin/mode checks.
3. [`safe_mppi/environment.py`](safe_mppi/environment.py) — dynamics, collision clearance, and goal.
4. [`safe_mppi/geometry.py`](safe_mppi/geometry.py) — uniform triangular polytope and `H_P`.
5. [`safe_mppi/controller.py`](safe_mppi/controller.py) — mode-1 sampling, costs, and safety filter.
6. [`safe_mppi/acquire.py`](safe_mppi/acquire.py) — per-gamma execution, NPZ schema, and metrics.
7. [`safe_mppi/visualize.py`](safe_mppi/visualize.py) — the two reproducible public figures.
8. [`safe_mppi/expansion.py`](safe_mppi/expansion.py) — explicit future-stage boundary.

Run the semantic tests with:

```bash
python -m pytest -q
```

## Scope and limitations

- The robot is a point mass; real vehicle radius, tracking error, and hardware latency are absent.
- Obstacles are static spheres or full-height vertical cylinders.
- The standalone package implements the online nominal `H_P` filter, not the separate 3D SOCP
  verifier/`valid2` pipeline shown in GREEN in the imported research GIFs.
- The 80x9 learned encoder, pretraining, safe flow expansion, adaptive gamma, wind robustness, and
  moving-obstacle prediction are not implemented here.
- The platform authority is recorded for downstream work, while this demonstration controller is
  intentionally capped at `1.2 m/s^2`.
- One paired seed per gamma is sufficient to reproduce the supplied visuals, not to support a
  general performance claim.
