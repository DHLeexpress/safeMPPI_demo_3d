# Minhyuk three-sphere clutter deployment handoff

Deployment-only package for one known three-sphere map. The model has no
onboard perception: rebuild this handoff with a new concrete config whenever
the obstacle map changes. The randomized template is provenance only and must
never be passed to a deployment runner.

- selected expansion round: `10`
- selection criterion: Among evaluated positive rounds {1,10,20,30,40,50}: maximize disjoint randomized-sphere pooled SR; tie-break by window validity, then earliest round.
- pretrained SHA-256: `b0fb3313e9bed30135517aa81c4a8677e070b31d3b6fa4b6bdb986b08952a4d6`
- expanded SHA-256: `657210c7b87716b8fa7da8121e00bab6bf5adc173383338431e2168df3566c5f`
- concrete config SHA-256: `2b54f98a30657cd0da0352a16279d7b72cb8e70cb1a2ce246b99718a0947628e`
- exact-map fixed-scene evaluation SHA-256: `068a3d5f3906c3521fa585f0e2364746c99dee4ed7ac1acc7e4c2cf5c89912a9`
- checkpoint output: raw pre-smoothing acceleration; governor/tracking external
- status: experimental, not flight-safety-qualified

## Deterministic successful fixed-scene examples

These are the first `SUCCESS` rows in evaluator seed order, without manual
trajectory curation. `null` means that the unbiased fixed-scene M-rollout bank
contained no successful example for that model/gamma.

| gamma | pretrained r0 seed | expanded r010 seed |
|---:|---:|---:|
| 0.1 | `1091077` | `1091003` |
| 0.3 | `1101047` | `1101010` |
| 0.5 | `1111017` | `1111017` |
| 1 | `1121283` | `1121024` |

### Pretrained round 0

```bash
python scripts/export_lab_flow_frozen_references.py \
  --config "flow_deployment/minhyuk_clutter_handoff/concrete_three_sphere_config.json" \
  --expected-config-sha256 2b54f98a30657cd0da0352a16279d7b72cb8e70cb1a2ce246b99718a0947628e \
  --checkpoint "flow_deployment/minhyuk_clutter_handoff/pretrained_visual_clutter_hp3d.pt" \
  --expected-checkpoint-sha256 b0fb3313e9bed30135517aa81c4a8677e070b31d3b6fa4b6bdb986b08952a4d6 \
  --sampling-temperature 1.0 \
  --gammas 0.1 0.3 0.5 1 \
  --seeds 1091077 1101047 1111017 1121283 \
  --output outputs/minhyuk_clutter_pretrained_validated_successes
```

### Expanded round 10

```bash
python scripts/export_lab_flow_frozen_references.py \
  --config "flow_deployment/minhyuk_clutter_handoff/concrete_three_sphere_config.json" \
  --expected-config-sha256 2b54f98a30657cd0da0352a16279d7b72cb8e70cb1a2ce246b99718a0947628e \
  --checkpoint "flow_deployment/minhyuk_clutter_handoff/expanded_visual_clutter_r010.pt" \
  --expected-checkpoint-sha256 657210c7b87716b8fa7da8121e00bab6bf5adc173383338431e2168df3566c5f \
  --sampling-temperature 1.0 \
  --gammas 0.1 0.3 0.5 1 \
  --seeds 1091003 1101010 1111017 1121024 \
  --output outputs/minhyuk_clutter_expanded_validated_successes
```


## Frozen/open-loop reference export

The generator replans against its simulated reference state; the resulting NPZ
is the frozen/open-loop artifact. The seed `91000` below is syntax-only and is
not claimed to be successful; use the validated commands above for successful
examples.

```bash
python scripts/export_lab_flow_frozen_references.py \
  --config "flow_deployment/minhyuk_clutter_handoff/concrete_three_sphere_config.json" \
  --expected-config-sha256 2b54f98a30657cd0da0352a16279d7b72cb8e70cb1a2ce246b99718a0947628e \
  --checkpoint "flow_deployment/minhyuk_clutter_handoff/expanded_visual_clutter_r010.pt" \
  --expected-checkpoint-sha256 657210c7b87716b8fa7da8121e00bab6bf5adc173383338431e2168df3566c5f \
  --sampling-temperature 1.0 \
  --gammas 0.1 0.3 0.5 1.0 \
  --seeds 91000 \
  --output outputs/minhyuk_clutter_frozen_r010
```

Inspect `manifest.json`; process exit alone does not establish success.

## Native closed-loop state-feedback smoke

```bash
python scripts/run_lab_flow_deployment.py \
  --config "flow_deployment/minhyuk_clutter_handoff/concrete_three_sphere_config.json" \
  --expected-config-sha256 2b54f98a30657cd0da0352a16279d7b72cb8e70cb1a2ce246b99718a0947628e \
  --checkpoint "flow_deployment/minhyuk_clutter_handoff/expanded_visual_clutter_r010.pt" \
  --expected-checkpoint-sha256 657210c7b87716b8fa7da8121e00bab6bf5adc173383338431e2168df3566c5f \
  --sampling-temperature 1.0 \
  --gamma 0.3 \
  --seed 91000 \
  --output outputs/minhyuk_clutter_closed_loop_r010
```

This exercises the unchanged native harness offline; it is not a hardware
flight command or safety certificate. Verify `deployment_contract.json`.
Aggregate clearance covers all three spheres, but the unchanged `deploy_sim`
log/GIF displays only the first sphere and has no online obstacle-collision
abort.

## Evidence

[`evidence/`](evidence/) contains the cylinder demonstration overlay,
pretraining qualification, disjoint randomized-domain curves, fixed-scene
temperature-one gallery, and the actual committed-success mechanism
visualizations. The three committed trajectories shown per
\((\mathrm{round},\gamma)\) come from separate randomized scenes; only the
fixed-scene gallery is evidence about conditional multimodality on one map.

## Live hardware controller contract

The repository has no live-flight CLI. Hardware integration must construct the
same authenticated controller and pass `[p_meas, v_ref]` at every 10 Hz replan:

```python
from flow_deployment.lab_pretrained import (
    load_lab_deployment_controller,
    sha256_file,
)
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment

config_path = "flow_deployment/minhyuk_clutter_handoff/concrete_three_sphere_config.json"
if sha256_file(config_path) != "2b54f98a30657cd0da0352a16279d7b72cb8e70cb1a2ce246b99718a0947628e":
    raise ValueError("concrete deployment config SHA-256 mismatch")
cfg = load_config(config_path)
env = TaskEnvironment(cfg)
controller, contract = load_lab_deployment_controller(
    "flow_deployment/minhyuk_clutter_handoff/expanded_visual_clutter_r010.pt",
    env,
    sampling_temperature=1.0,
    expected_sha256="657210c7b87716b8fa7da8121e00bab6bf5adc173383338431e2168df3566c5f",
)
action, info = controller.plan(
    state=[px, py, pz, vx_ref, vy_ref, vz_ref],
    goal=env.goal,
    gamma=0.3,
    seed=episode_seed * 100_000 + step,
)
```

To use the byte-identical pretrained baseline instead, replace the checkpoint
with `pretrained_visual_clutter_hp3d.pt` and its SHA-256 `b0fb3313e9bed30135517aa81c4a8677e070b31d3b6fa4b6bdb986b08952a4d6`. Apply smoothing and
the reference governor exactly once outside the policy.
