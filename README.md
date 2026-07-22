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
| Task-agnostic B1 Safe Flow Expansion core | implemented | [`expansion.py`](safe_mppi/expansion.py), [`flow_model.py`](safe_mppi/flow_model.py) |
| Expansion result/gallery/video skeletons | implemented | [`expansion_visualize.py`](safe_mppi/expansion_visualize.py) |

The expansion core is deliberately separated from task facts. A new 3D task must provide its own
context, dynamics, nominal `H_P` gate, full-H verifier, and execution cost through the documented
adapter. The core does not pretend that the current nominal-polytope controller is a full verifier.

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --config default_config.json --output outputs/default_run --device cpu

# Requested biased 20-inch-ball demonstrations (8 paired seeds x 4 gammas)
python run.py --config configs/ball_biased_demo.json \
  --output outputs/ball_biased_demo --device cpu
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
an exponential altitude penalty `w exp((z-2)/T)` that keeps every rollout below the ball's
latitude-0 circle while the seed family fans out across passage angles, a `1 m/s^2`
demonstration cap, and a `[0.1,0,0]` warm-start bias. Analyze a finished run with
`python -m safe_mppi.ball_analysis --run <output_dir>` and render the animated rollouts with
`python -m safe_mppi.ball_gif --run <output_dir>`.

All ten seeds per gamma with the evolving translucent nominal polytope (camera orbits so the
level bands are seen from different angles); right panel is the head-on view from the start:

| gamma 0.1 | gamma 0.3 |
|---|---|
| ![ball-below gamma 0.1](docs/assets/ball_below/ball_evolve_g0.1.gif) | ![ball-below gamma 0.3](docs/assets/ball_below/ball_evolve_g0.3.gif) |

| gamma 0.5 | gamma 1.0 |
|---|---|
| ![ball-below gamma 0.5](docs/assets/ball_below/ball_evolve_g0.5.gif) | ![ball-below gamma 1.0](docs/assets/ball_below/ball_evolve_g1.gif) |

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

## Biased 20-inch-ball demonstration task

[`configs/ball_biased_demo.json`](configs/ball_biased_demo.json) is the small-data task requested for
the next expansion experiment. “20-inch ball” is interpreted explicitly as a 20-inch **diameter**:
the sphere radius is `0.254 m`.

| item | fixed value |
|---|---:|
| start / goal | `(0,0,2)` m / `(3,0,2)` m |
| sphere | center `(1.5,0,2)` m, radius `0.254 m` |
| gamma | `.1,.3,.5,1` |
| paired demonstration seeds | `0,...,7` for every gamma |
| demonstration acceleration cap | `1 m/s^2` per axis |
| initial warm action | `(0.1,0,0) m/s^2` |
| horizon / MPPI samples | `10 / 1536` |
| running / terminal / control / smoothness | `5 / 20 / .05 / .8` |
| proximity / progress auxiliary costs | `0 / 0` |
| lower-half bias | `.05 exp((z-2)/.012)` per predicted state |

Centroid steering is disabled (`centroid_gain=urgency_floor=0`, `sigma_aniso=1`). Thus the only
ranking terms are running state error, terminal state error, control effort/smoothness, and the
declared lower-half exponential bias. No proximity preference or hidden obstacle cost is used.

The checked-in eight-seed result is fully successful and collision-free. Near the ball
(`1.1 <= x <= 1.9`), every saved trajectory stays below the equatorial plane; the largest observed
`z` is `1.943 m`. Reproduce that predicate with `python scripts/audit_ball_demo.py`. The empirical
averages are:

| gamma | SR | CR | minimum clearance [m] | time-to-goal [s] | control variation | below-plane fraction |
|---:|---:|---:|---:|---:|---:|---:|
| .1 | 1.00 | 0 | .258 | 4.46 | .811 | .958 |
| .3 | 1.00 | 0 | .115 | 4.33 | .747 | .977 |
| .5 | 1.00 | 0 | .130 | 4.74 | .753 | .976 |
| 1 | 1.00 | 0 | .113 | 4.21 | .779 | .976 |

The strongest qualitative predictions hold: gamma `.1` keeps the largest clearance, gamma `1` is
fastest, and gamma `.5` has lower control variation than gamma `1`. Adjacent gamma averages are not
strictly monotone; the code does not add gamma-specific costs to manufacture that ordering.

| all paired rollouts | paired rollouts separated by gamma |
|---|---|
| ![Biased ball rollouts](examples/ball_biased_demo/gamma_rollouts_3d.png) | ![Per-gamma biased ball rollouts](examples/ball_biased_demo/gamma_rollouts_by_gamma.png) |

![Nominal polytopes and H_P levels](examples/ball_biased_demo/nominal_levelsets_by_gamma.png)

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
repository generates static PNG figures, not those detailed three-panel animations.

## Standalone B1 Safe Flow Expansion

The implementation in [`safe_mppi/expansion.py`](safe_mppi/expansion.py) is a single AFE arm: no
expert fallback, demo replay, proximal leash, curriculum, recovery starts, rollback, or evaluation
temperature search. [`ConditionalFlowMLP`](safe_mppi/flow_model.py) is a lightweight default, but a
task can replace it while preserving the same expansion loop.

For candidate plan `U` and context `c`, the current policy supplies a representation
`phi_s(U,c)`. Normalize the representations and initialize the RBF length scale from 50 pretrained
samples:

```text
ell = mean_{i<j} || normalize(phi_i) - normalize(phi_j) ||_2
k(z,z') = exp(-||z-z'||^2 / (2 ell^2))
sigma_n^2(z) = k(z,z) - k(z,Z+) [K(Z+,Z+) + lambda I]^{-1} k(Z+,z).
```

`Z+` contains only full-H verifier positives from the previous `W` rounds. It is capped and sampled
without replacement with equal round/gamma opportunity. At each active context, sample `K` plans
and acquire `B` sequentially:

```text
pi(j | pending) proportional to exp((sigma_j - max sigma) / beta).
```

Every successfully evaluated selected-B query enters `D`; only full-H positives enter `D+` and the
next-round GP. Among plans satisfying both the full-H verifier and the separate one-step nominal
`H_P` gate, execute the plan with the smallest task-supplied SafeMPPI cost. Execute only its first
action and replan. No eligible plan means `NVP`, and that episode terminates fail-closed.

After a round, recent `D+` replay has equal mass over
`gamma -> (round, episode) -> context -> positive query`. With `inner_steps=None`, every eligible
positive is used exactly once without replacement. Optional negative gradients use

```text
g = g_positive - rho g_negative,
rho = alpha ||g_positive|| / (||g_negative|| + eps).
```

`alpha=0` is exactly the positive-only update. NVP-context queries are the negative population; they
do not become positive GP observations.

`checkpoint_000.pt` is always the untouched pretrained policy. Updates produce
`checkpoint_001.pt,...,checkpoint_N.pt`; result curves and galleries must evaluate those saved
checkpoints with a separate, untilted raw-policy seed bank.

### Variables to change first

The “starter” column is intentionally cheap. The B1 reference column records the scale used in the
2D study; it is not a claim that those values transfer to a new 3D verifier.

| variable | starter default | B1 reference / meaning |
|---|---:|---|
| rounds | `10` | start with `10`; extend only after raw evaluation |
| parallel episodes per gamma | `2` | `8`; independent synchronous replicas preserve route support |
| `K` generated plans | `16` | `16` |
| `B` verifier queries | `4` | `4`; all successful query results enter `D` |
| batch size | `32` | `128` |
| inner steps | `None` | exact one-pass; `ceil(|eligible D+|/batch)` |
| learning rate | `3e-5` | B1 used `1e-5`; reduce before adding many passes |
| replay window `W` | `2` rounds | `2` rounds |
| RBF GP cap | `256` positives | `512` or `768`; affects GP cost, not `D+` replay |
| GP noise `lambda` | `1e-2` | `1e-2` |
| RBF length scale | required calibration | mean pairwise distance of 50 pretrained embeddings |
| beta | `.05` smoke value | calibrate once on representative pools, then freeze |
| adaptive beta | **false** | opt-in only; `true` retargets ESS after each round |
| ESS target | `.5` | used by `calibrate_fixed_beta`; it is not a validity probability |
| negative alpha | `0` | try `.001` or `.01` only with an audited NVP definition |
| expert/demo fraction | `0` | remains `0` for self-generated expansion |

For a fixed initial beta, collect representative uncertainty score pools, call
`calibrate_fixed_beta(pools, target=.5)`, write the returned scalar into `ExpansionConfig.beta`, and
leave `adaptive_beta=False`. Do not choose beta from the absolute magnitude of sigma alone.

### Porting to another 3D task

Implement the five methods in `ExpansionTask`: `reset`, `context`, `verify`, `advance`, and
`terminal`. The verifier returns `Verification(valid, hp_eligible, margin, execution_cost, error)`.
That is the only place where task-specific dynamics, a 3D nominal polytope, SOCP geometry, and the
native SafeMPPI cost belong. Supply 50 pretrained embeddings or an explicitly justified RBF length
scale, then call `run_safe_expansion`.

[`expansion_visualize.py`](safe_mppi/expansion_visualize.py) provides three task-neutral outputs:

- `plot_expansion_results`: result curves from `metrics.jsonl`;
- `plot_rollout_gallery`: overlaid raw-policy trajectories by round and gamma; and
- `render_expansion_mechanism`: K plans, selected-B uncertainty, verifier positives, executed first
  action, accumulated positive/NVP states, and a `sigma(phi_s)` colorbar.

Pass an `event_callback` to `run_safe_expansion` to receive the state before/after replanning, all K
candidates and marginal uncertainties, selected B indices, verifier outputs, and the executed index.
The task adapter converts candidate controls to 3D paths and nominal/verifier polytope vertices
before constructing `MechanismFrame`; the generic core does not guess that geometry.

Reference mechanism video: [B1 rounds 0/5/10/15](docs/assets/b1_expansion_mechanism_reference.mp4).

### Ball-task deployment: pretraining + expansion + representation audit

[`docs/BALL_FLOW_EXPANSION.md`](docs/BALL_FLOW_EXPANSION.md) deploys this loop end to end on the
20-inch-ball task with the 10-D context `c_t = [g-p, v, b_near-p, gamma]` and a 30-D plan flow
(`scripts/pretrain_ball_flow.py`, `scripts/run_ball_expansion.py`,
`scripts/evaluate_ball_expansion.py`, `safe_mppi/ball_flow_diagnostics.py`). It reports raw
temperature-1 success/collision, above/below/left/right route coverage, untilted GREEN-verifier
validity, gamma trends against the SafeMPPI demonstrator, sigma-tilted acquisition anatomy, and
an automated fixed-bank audit of the penultimate noised representation (t-SNE panels, local
probes, and the high-sigma new-mode discovery rate). Curated figures live in
[`docs/assets/ball_flow/`](docs/assets/ball_flow/).

![Mode presence: tilted acquisition vs raw sampling](docs/assets/ball_flow/mode_timeline.png)
It is a format reference, not evidence for this new 3D ball task.

## Code reading order

1. [`default_config.json`](default_config.json) — the entire public experiment contract.
2. [`safe_mppi/config.py`](safe_mppi/config.py) — typed loading and strict zero-margin/mode checks.
3. [`safe_mppi/environment.py`](safe_mppi/environment.py) — dynamics, collision clearance, and goal.
4. [`safe_mppi/geometry.py`](safe_mppi/geometry.py) — uniform triangular polytope and `H_P`.
5. [`safe_mppi/controller.py`](safe_mppi/controller.py) — mode-1 sampling, costs, and safety filter.
6. [`safe_mppi/acquire.py`](safe_mppi/acquire.py) — per-gamma execution, NPZ schema, and metrics.
7. [`safe_mppi/visualize.py`](safe_mppi/visualize.py) — reproducible rollout and level-set figures.
8. [`safe_mppi/flow_model.py`](safe_mppi/flow_model.py) — lightweight task-sized conditional flow.
9. [`safe_mppi/expansion.py`](safe_mppi/expansion.py) — task-neutral B1 expansion loop.
10. [`safe_mppi/expansion_visualize.py`](safe_mppi/expansion_visualize.py) — result/gallery/video skeletons.

Run the semantic tests with:

```bash
python -m pytest -q
```

## Scope and limitations

- The robot is a point mass; real vehicle radius, tracking error, and hardware latency are absent.
- Obstacles are static spheres or full-height vertical cylinders.
- The current SafeMPPI data collector implements online nominal `H_P`, not the separate 3D SOCP
  verifier shown in GREEN. Expansion is runnable only after a task adapter supplies that verifier.
- No trained checkpoint for the 3D ball task is claimed yet. The lightweight flow class and
  expansion engine are infrastructure for the next experiment, not a fabricated result.
- The 80x9 learned encoder, adaptive gamma, wind robustness, and moving-obstacle prediction are not
  implemented here.
- The platform authority is recorded for downstream work, while this demonstration controller is
  intentionally capped at `1.2 m/s^2`.
- One paired seed per gamma is sufficient to reproduce the supplied visuals, not to support a
  general performance claim.
