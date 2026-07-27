# Minhyuk handoff: frozen lab flow policies

This directory contains two **pretrained, unexpanded** lab-frame policies. The
visual policy is the intended deployment candidate; `raw10` remains an
authenticated baseline because the visual model is not uniformly better at
every \(\gamma\).

| item | visual candidate | raw baseline |
|---|---|---|
| checkpoint | `pretrained_visual_hp3d.pt` | `pretrained_raw10.pt` |
| SHA-256 | `fc4d215817b56d74730a0a90f6abc57d17dbeb7626302add535760399cdeeeb4` | `cdc27062cdf60e3faf54ed5a7e6dd23e0c45e8208343ff00b9455a799034d059` |
| context | \([g-p,v,\gamma]\) + 3-D safety grid | \([g-p,v,b_{\rm near}-p,\gamma]\) |
| flow trunk | \(70\to48\to32\to30\), SiLU | \(41\to48\to32\to30\), SiLU |
| output | raw \(H=10\), 3-D acceleration window | same |
| action limit | \(0.3\ {\rm m/s^2}\) per component | same |
| default latent scale | \(\tau=1\) | \(\tau=1\) |

The visual grid has shape \(3\times16\times12\times12\) with channels
`occupancy`, `nominal_polytope_mask`, and clipped \(H_P\). A compact Conv3D
encoder maps it to 32 dimensions. The grid is robot-centered and world-axis
aligned. It is rebuilt from the current position and the **known configured
map**; this is not an onboard-perception claim.

The policy predicts the pre-smoothing raw acceleration command. Minhyuk's
unchanged deployment harness owns smoothing, reference integration, geofence
checks, and vehicle interface. Do not duplicate those operations in the
policy. Sampling temperature means

\[
x_0\sim\mathcal N(0,\tau^2 I).
\]

## 1. Online state-feedback inference

```bash
python scripts/run_lab_flow_deployment.py \
  --checkpoint flow_deployment/minhyuk_handoff/pretrained_visual_hp3d.pt \
  --expected-checkpoint-sha256 \
    fc4d215817b56d74730a0a90f6abc57d17dbeb7626302add535760399cdeeeb4 \
  --sampling-temperature 1.0 \
  --gamma 0.3 \
  --seed 91000 \
  --output outputs/minhyuk_visual_online
```

For Minhyuk's hardware runner, instantiate the identical controller:

```python
from flow_deployment.lab_pretrained import load_lab_deployment_controller
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment

cfg = load_config("configs/lab_ball_pretrain.json")
env = TaskEnvironment(cfg)
controller, contract = load_lab_deployment_controller(
    "flow_deployment/minhyuk_handoff/pretrained_visual_hp3d.pt",
    env,
    sampling_temperature=1.0,
    expected_sha256=(
        "fc4d215817b56d74730a0a90f6abc57d17dbeb7626302add535760399cdeeeb4"
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

`action` is acceleration feedforward, not a motor command. The present
`deploy_sim` harness supplies measured position \(p_{\rm meas}\) but its
reference velocity \(v_{\rm ref}\), not measured velocity, in this state.

## 2. Frozen trajectory export

```bash
python scripts/export_lab_flow_frozen_references.py \
  --checkpoint flow_deployment/minhyuk_handoff/pretrained_visual_hp3d.pt \
  --expected-checkpoint-sha256 \
    fc4d215817b56d74730a0a90f6abc57d17dbeb7626302add535760399cdeeeb4 \
  --sampling-temperature 1.0 \
  --gammas 0.1 0.3 0.5 1.0 \
  --seeds 91000 \
  --output outputs/minhyuk_visual_frozen
```

Each NPZ contains `dense_positions`, `states`, raw `controls`, governed
`executed_controls`, and `dense_steps`.

## Qualification and limits

Both models were audited using the same raw temperature-1 \(M=50/\gamma\)
protocol. The visual model's pooled SR is 51% versus 44% for raw10, and pooled
OOB is 17% versus 29%. Its pooled CR is worse, 32% versus 27%, driven by
\(\gamma=1\) (48% CR). It discovered no above route in this audit. Exact
per-\(\gamma\) values are in `pretrain_visual_manifest.json` and
`pretrain_manifest.json`.

The visual seed-91000 frozen export succeeds at all four gammas. Under the
current parameter-free native vehicle, however, the \(\gamma=.3\), seed-91000
online smoke does **not** qualify: it stops near the goal with a frozen-state
soft abort and enters the configured safety sphere by 0.065 m. The inference
interface works, but the online controller is not deployment-qualified. These
are reproducibility and software-interface checks—not a flight safety
certificate. The checkpoint has no online flow expansion, verifier guarantee,
onboard map estimator, or hardware tracking guarantee.
