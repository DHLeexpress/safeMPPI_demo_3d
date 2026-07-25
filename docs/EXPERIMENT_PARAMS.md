# Experiment parameters — quadrotor flight trials

Parameters used to run `Mode1SafeMPPI` on a real quadrotor, and the measurements
that motivated each change. **The controller in `safe_mppi/` is unmodified** —
everything here is configuration, so a new controller drop-in can reuse it.

Task config: [`configs/crazyflie_mppi_corner.json`](../configs/crazyflie_mppi_corner.json).
The mocap/radio tooling is machine-specific and is intentionally not part of this
repository.

## Task

| item | value |
|---|---|
| taskspace | origin `(-2.5, -1.7, 0.0)`, size `3.8 x 3.5 x 1.5` m |
| start | `(-2.1, 1.5, 0.6)` |
| goal | `(0.7, -1.5, 1.2)` |
| obstacle | sphere `(-0.541, -0.416, 1.217)`, radius `0.379` m |
| reach radius | `0.35` m |
| gamma | `0.3` |

The obstacle is a 0.359 m-diameter ball measured in situ (radius `0.179`); the
config radius adds the vehicle half-width (`0.05`) and a tracking-error margin
(`0.15`), because the package fixes `robot_radius = obstacle_margin = 0` — so all
margin must be baked into the sphere radius.

## Changes from the package defaults

| parameter | default | used | why |
|---|---:|---:|---|
| `demo_u_max` | 1.2 | **0.3** | 1.2 produced ~2.7 m/s peaks in a 3.5 m arena |
| `noise_sigma` | 1.0 | **0.5** | narrower sampling; smoother commands |
| `smooth_weight` | 0.12 | **0.35** | penalise command jerk |
| `control_weight` | 0.03 | **0.05** | " |
| `soft_clearance_target` | 0.10 | **0.30** | keep clear of the ball by more than the tracking error |
| `soft_clearance_weight` | 25 | **60** | " |
| `z_bias_weight` | 0 | **2.0** | discourage climbing (see below) |
| `z_bias_plane` | 2.0 | **1.35** | " |
| `z_bias_temperature` | 0.05 | **0.12** | " |
| `reach_radius` | — | **0.35** | must exceed the vehicle's tracking error |

A non-standard top-level `safety` block (ignored by `load_config`, which keeps
unknown keys in `cfg.raw`) records the flight-volume limits:

```json
"safety": { "safe_min": [-2.5, -1.7, 0.4], "safe_max": [1.3, 1.8, 2.0] }
```

## Measurements that constrain the controller

These came out of the flight trials and are worth knowing before tuning:

1. **Acceleration is a feedforward, not a motor command.** MPPI's `u` is streamed
   as the acceleration term of a full-state setpoint, with the position/velocity
   terms obtained by integrating the double integrator one step
   (`p = p + v t + ½ u t²`, `v = v + u t`) at ~100 Hz between 10 Hz replans.

2. **Tracking error ~0.16 m.** On one run the vehicle passed 0.162 m from the
   goal while `reach_radius` was 0.15 m, so the success test never fired and it
   flew on past. Any arrival tolerance must exceed the tracking error.

3. **Goal overshoot 0.2–0.3 m is inherent** and did *not* improve with more
   acceleration authority (swept `demo_u_max` 0.3/0.5/0.8/1.2 on a
   momentum plant). Keep goals at least ~0.4 m clear of any flight-volume limit.

4. **The altitude loop is underdamped.** With the onboard position gains
   (`kp_z = 1.25`, `kd_z = 0.4`) the damping ratio is
   `ζ = kd / (2√kp) ≈ 0.18`, which predicts ~56 % overshoot and 2.86 s to peak;
   measured 77 % and 3.01 s. A useful rule:

   > altitude overshoot ≈ v_climb / ω_n,  ω_n = √kp_z ≈ 1.12 rad/s

   A 0.56 m/s climb therefore overshoots ~0.5 m. Capping *vertical* speed is far
   more effective than capping total speed. Planning a path that goes **around**
   an obstacle rather than **over** it avoids the problem entirely.

5. **Plan inside a tighter box than the safety limit.** The barrier is reactive:
   the controller will happily plan up to the taskspace bound, and overshoot then
   carries it past. Here the taskspace ceiling is 1.5 m while the flight-volume
   ceiling is 2.0 m.

## Reproducing in simulation

```bash
python run.py --config configs/crazyflie_mppi_corner.json \
  --output outputs/mppi_corner --device cpu
```

Note that a kinematic simulator that tracks position setpoints exactly cannot
reproduce items 2–5 above: it has no momentum, no tracking lag and no
measurement noise, so it reports success on runs that fail on hardware.
