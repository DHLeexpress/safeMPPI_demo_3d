# `deploy_sim` — test a controller the way it is actually flown

Planning in a clean geometric simulator is not what breaks real flights. What
breaks them is the deployment layer: a finite replan rate, setpoint streaming,
geofences, state-estimate dropouts, and arrival tolerances that are smaller than
the tracking error.

This package runs a controller through **exactly the loop used on the real
vehicle**. Only the vehicle differs between this and a real flight -- and here
the vehicle simply follows the streamed setpoint, which hardware does to within
~30 mm (see below).

**It needs nothing beyond `requirements.txt`** — no simulator install, no ROS,
no motion capture, no vehicle.

```bash
python deploy_sim/run_offline.py --config configs/crazyflie_mppi_corner.json
```

Outputs land in `outputs/deploy_sim/`: a CSV/NPZ log (measured **and** commanded
state, so tracking error is directly measurable) and figures showing the
trajectory, the nominal polytope with its `H_P` level sets, and altitude.
Add `--gif` for an orbiting animation.

## Testing your own controller

Implement two methods:

```python
class MyController:
    def __init__(self, cfg, env, device="cpu"):   # cfg = the "safemppi" config block
        ...
    def reset(self):                              # called once per run
        ...
    def plan(self, state, goal, gamma, seed=0):
        # state : (6,) [x, y, z, vx, vy, vz]
        # goal  : (3,)
        # return: (3,) commanded ACCELERATION [m/s^2], and an info dict
        return u, {}
```

then

```bash
python deploy_sim/run_offline.py --controller mymodule:MyController
```

[`example_controller.py`](example_controller.py) is a working template (a naive
PD law). Run it and compare — it reaches the goal but cuts **inside** the
obstacle safety sphere, which is the sort of thing this harness is for.

`gamma` is the barrier-contraction parameter used by the bundled SafeMPPI;
ignore it if your method has no equivalent.

## The acceleration is a feedforward, not a motor command

Each plan is converted to a kinematically consistent full-state setpoint and
streamed at ~100 Hz between 10 Hz replans:

```
v_ref <- v_ref + u·dt        p_ref <- p_ref + v_ref·dt        a_ref = u
cmdFullState(p_ref, v_ref, u, yaw=0, omega=0)
```

The vehicle's onboard position controller tracks that setpoint. So `u` never
goes to motors; it is the acceleration feedforward. A reference governor keeps
`p_ref` within `--max-lead` of the measured position so the setpoint can never
run away.

## The vehicle model: it follows the setpoint

`vehicle.py` has **no fitted parameters**. In simulation the vehicle is exactly
where the streamed setpoint says it is.

That is a measurement, not an assumption. Across four live-SafeMPPI hardware
flights (Crazyflie 2.1 + Vicon, gamma 0.1/0.3/0.5/1.0, 343 control cycles), the
measured position tracked the streamed reference to:

```
27-31 mm RMSE,  52 mm worst case,  +21 mm systematic altitude bias
```

An order of magnitude below the tolerances that decide success here (arrival
radius, obstacle margin, geofence margin). The residual is deliberately **not**
simulated -- a simulated residual becomes a number you start trusting. Treat
`RESIDUAL_MAX_M = 0.052` as a margin budget: a trajectory that leaves less room
than that against the floor, the fence or an obstacle is too tight to call safe
from simulation alone.

**Validity.** Measured at <=0.91 m/s and <=0.50 m/s^2 commanded -- about 9% of
the vehicle's lateral acceleration authority. Near the thrust limit, or for
direction changes faster than the lateral position loop (omega_n ~ 0.63 rad/s),
the vehicle can no longer match the setpoint and this becomes optimistic.

### Why the old "calibrated plant" was deleted (2026-07-26)

`plant.py` modelled the onboard Mellinger loop, thrust limits, estimator lag and
mocap noise. Its three fitted parameters (`est_tau=0.08`, `noise=0.0017`,
`rate_scale=1.111`) came from a single flight recorded **before a control-loop
timing bug was fixed**. In that loop each substep advanced the reference by a
nominal `dt_sub` while ~1.25x that elapsed in wall time, so `p_ref` and `v_ref`
contradicted each other and the onboard controller chased a velocity the
reference never had. `est_tau` absorbed that as if it were vehicle lag.

After the fix it predicted **176-242 mm** of tracking error where the vehicle
actually showed **27-31 mm** -- 5.7-8.2x pessimistic -- and it wrongly condemned
gamma 0.5 and gamma 1.0 as collisions. Both flew with clearance to spare.

A model that wrong is worse than none, because it silently rejects good
controllers. If a dynamics model is needed again, fit a new one to post-fix data
and give it a new name. `characterise_plant.py` and its figure were removed with
it.

## Safety layers (identical to the flight configuration)

| layer | trigger | response |
|---|---|---|
| soft geofence | leaves the flight volume | stop and land in place |
| hard geofence | leaves it by a further margin | emergency stop |
| over-speed | > 1.8 × cap | land |
| runaway speed | > 2.5 × cap | emergency stop |
| frozen/invalid state estimate | no change for `--state-timeout` (0.3 s) | land |
| setpoint clamp | always | commanded position kept inside the fence |

The frozen-estimate check earns its place: a stale pose looks perfectly healthy
— finite, inside the fence, reporting zero finite-difference velocity — while
the vehicle keeps flying. Test your controller against it:

```bash
python deploy_sim/run_offline.py --fault-freeze-at 3.0
```

## Reading the summary

```
outcome                          reached goal
closest_to_goal_m                0.241     # must exceed nothing; but the arrival
                                           # tolerance must exceed tracking error
clearance_beyond_safety_sphere_m 0.026     # <0 means inside the configured shell
peak_speed_mps                   1.03      # achieved, vs the commanded cap
fence_margin_m                   0.154     # <0.052 ⇒ inside the measured tracking residual
peak_z_m                         1.143
```

Two traps worth knowing, both learned the hard way:

1. **Arrival tolerance must exceed tracking error.** A run once passed 0.162 m
   from the goal with a 0.15 m tolerance, so "reached" never fired and the
   vehicle flew on past and out of the fence.
2. **Differentiate over ~0.1 s, not sample-to-sample.** At 100 Hz with
   millimetre noise, a raw finite difference reports roughly double the true
   speed. `summarize()` already does this; do the same in your own analysis.

## Geofence

Taken from an optional top-level `"safety"` block in the task config:

```json
"safety": { "safe_min": [-2.5, -1.7, 0.4], "safe_max": [1.3, 1.8, 2.0] }
```

`load_config` ignores unknown top-level keys, so this is safe to add. Without
it, the fence is derived from the taskspace. Note the barrier is *reactive*: the
controller will plan right up to the taskspace bound and overshoot carries it
past, so keep the **taskspace** ceiling below the **fence** ceiling.

See [`../docs/EXPERIMENT_PARAMS.md`](../docs/EXPERIMENT_PARAMS.md) for the
parameters used in the real trials and the measurements behind them.
