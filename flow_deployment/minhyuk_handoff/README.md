# Minhyuk handoff: frozen lab flow policy

This directory contains the best **completed** lab-frame pretrained policy
available on 2026-07-26. Fine-tuning is stopped. The checkpoint is not an
expanded policy.

## Frozen model

| item | value |
|---|---|
| checkpoint | `pretrained_raw10.pt` |
| SHA-256 | `cdc27062cdf60e3faf54ed5a7e6dd23e0c45e8208343ff00b9455a799034d059` |
| context | \([g-p,\;v,\;b_{\rm near}-p,\;\gamma]\in\mathbb R^{10}\) |
| flow network | \(41\to48\to32\to30\), SiLU, NFE \(=16\) |
| output | raw \(H=10\), 3-D acceleration window |
| action limit | \(0.3\ {\rm m/s^2}\) per component |
| deployment temperature | \(\tau=1\) by default, runtime configurable |

Here “temperature” means the flow base distribution

\[
x_0\sim\mathcal N(0,\tau^2 I).
\]

The model predicts the **pre-smoothing raw command**. Minhyuk's unchanged
deployment harness owns command smoothing, reference integration, geofence
checks, and the calibrated plant. These operations must not be duplicated in
the policy.

## 1. Online state-feedback inference

The following runs the policy at every 10 Hz deployment replan. The harness
passes its current measured plant position and current reference velocity to
the policy. The policy is therefore reconditioned on the latest position at
every call.

```bash
python scripts/run_lab_flow_deployment.py \
  --checkpoint flow_deployment/minhyuk_handoff/pretrained_raw10.pt \
  --sampling-temperature 1.0 \
  --gamma 0.3 \
  --seed 91000 \
  --output outputs/minhyuk_flow_online_g03
```

For Minhyuk's own hardware runner, instantiate the same controller:

```python
from flow_deployment.lab_pretrained import load_lab_deployment_controller
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment

cfg = load_config("configs/lab_ball_pretrain.json")
env = TaskEnvironment(cfg)
controller, contract = load_lab_deployment_controller(
    "flow_deployment/minhyuk_handoff/pretrained_raw10.pt",
    env,
    sampling_temperature=1.0,
    expected_sha256=(
        "cdc27062cdf60e3faf54ed5a7e6dd23e0c45e8208343ff00b9455a799034d059"
    ),
)

# At every 10 Hz replan:
action, info = controller.plan(
    state=[px, py, pz, vx, vy, vz],
    goal=env.goal,
    gamma=0.3,
    seed=episode_seed * 100_000 + step,
)
```

`action` is an acceleration feedforward command, not a motor command.

## 2. Frozen trajectory export

This reproduces one governed, goal-reaching frozen reference per gamma. Seed
91000 was fixed before export and succeeds for all four gammas at \(\tau=1\).

```bash
python scripts/export_lab_flow_frozen_references.py \
  --checkpoint flow_deployment/minhyuk_handoff/pretrained_raw10.pt \
  --sampling-temperature 1.0 \
  --gammas 0.1 0.3 0.5 1.0 \
  --seeds 91000 \
  --output outputs/minhyuk_frozen_references
```

Each NPZ contains `dense_positions`, `states`, raw `controls`, governed
`executed_controls`, and `dense_steps`. The exporter also writes a 3-D PNG/PDF
overlay and an authenticated manifest.

## Qualification and limits

The M=50-per-gamma raw audit is in `pretrain_manifest.json`. Pooled SR is
approximately 44%; this is the best completed lab pretrained checkpoint, not a
flight-ready safety claim. Runtime \(\tau=1\) is the default because the
completed temperature sweep found that lower temperatures reduced plant
performance.

This checkpoint is `raw10`, **not** the unfinished visual-encoder model. Current
plant position is used to rebuild the closest-obstacle-boundary context, but no
3-D visual token is produced. The visual training process was stopped before a
checkpoint was saved, so describing this model as visual would be incorrect.
The deployment API is state-feedback compatible, but a future visual checkpoint
must be separately trained, qualified, and authenticated.
