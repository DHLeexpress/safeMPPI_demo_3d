# Minhyuk handoff: Stage 2 canonical three-sphere baseline

This folder replaces the retired 20-inch/random-sphere r10 handoff. It contains
only the current fixed Stage-2 scene and an authenticated round-zero baseline.
Stage-2 expansion is still research-side work, so no provisional expanded
checkpoint is shipped. No file under `deploy_sim/` is changed.

## What is canonical

- start: `(-2.1, 1.5, 0.9)` m
- goal: `(0.7, -1.5, 0.9)` m
- geofence: `x=[-2.5,1.3]`, `y=[-1.7,1.8]`, `z=[0.4,2.0]` m
- three 15-inch-diameter spheres: physical radius `0.1905 m`
- robot inflation: `0.10 m`; modeled radius `0.2905 m`
- original, unflipped equilateral triangle; center spacing `0.70 m`
- canonical scene: the unjittered triangle in
  [`canonical_three_sphere_config.json`](canonical_three_sphere_config.json)
- research randomization law: every sphere independently moves only along its
  own start-goal-parallel line, with `sigma_parallel=0.30 m` and
  `sigma_perp=0`. This fixed handoff deliberately does not resample it.

The three modeled centers are:

| sphere | x | y | z | modeled radius |
|---:|---:|---:|---:|---:|
| 1 | -0.955869 | -0.238811 | 0.697927 | 0.2905 |
| 2 | -0.700000 | 0.000000 | 1.304145 | 0.2905 |
| 3 | -0.444131 | 0.238811 | 0.697927 | 0.2905 |

The modeled bodies have `0.119 m` pairwise surface gaps. The lower modeled
bodies clear the floor by only about `7.4 mm`; the physical spheres clear it by
about `0.107 m`. Longitudinal-only randomization does not change either height.

## Why this is an expansion baseline

Both current OOD tasks expose failures of the same cylinder-ID pretrained
policy:

| task | raw temperature-one bank | pretrained SR | CR | OOB | result after expansion |
|---|---:|---:|---:|---:|---|
| Stage 1, fixed single 20-inch sphere | M=20/gamma | 13.75% | 80.0% | 6.25% | r5 SR 98.75%, CR 0%, but all 79 successes use one route |
| Stage 2, canonical three 15-inch spheres | M=50/gamma | 48.5% | 49.5% | 2.0% | in progress; no expanded model shipped |

The Stage-2 per-gamma SR is `.46/.44/.48/.56` for
`gamma=.1/.3/.5/1.0`. Exact compact metrics are in
[`pretrained_r0_m50_summary.json`](pretrained_r0_m50_summary.json). The target
for Stage 2 is to move this raw policy toward 100% SR while preserving validity,
gamma trends, and genuinely different paths around/through the three-sphere
arrangement.

## Files to use

| purpose | file |
|---|---|
| fixed Stage-2 task and SafeMPPI recipe | [`canonical_three_sphere_config.json`](canonical_three_sphere_config.json) |
| pretrained HP100-T128-D3 policy | [`../minhyuk_stage1_handoff/checkpoints/hp100_t128_d3.pt`](../minhyuk_stage1_handoff/checkpoints/hp100_t128_d3.pt) |
| architecture/training contract | [`../minhyuk_stage1_handoff/model_contract.json`](../minhyuk_stage1_handoff/model_contract.json) |
| Stage-2 baseline metrics | [`pretrained_r0_m50_summary.json`](pretrained_r0_m50_summary.json) |
| all handoff hashes | [`SHA256SUMS`](SHA256SUMS) |

The checkpoint emits one raw `H=10` acceleration plan and uses a
robot-centered `1 x 32 x 32 x 100` clipped-`H_P` volume. It has no internal
governor or GRU. Deployment smoothing, measured-state feedback, and command
limits stay in Minhyuk's existing loop and must be applied exactly once.

## 1. Run native SafeMPPI on the canonical scene

This searches at most 64 fixed-scene controller-noise seeds for one
nominal-safe successful reference per gamma. It is a reference exporter, not
an unbiased SafeMPPI success-rate audit; inspect the attempt rows as well as
the accepted trajectories.

```bash
python run.py \
  --config flow_deployment/minhyuk_clutter_handoff/canonical_three_sphere_config.json \
  --output outputs/minhyuk_stage2_safemppi_one_per_gamma \
  --device cpu
```

Inspect `metrics.json` and the saved trajectories; process completion alone is
not a success claim.

## 2. Export frozen pretrained-policy trajectories

```bash
CFG=flow_deployment/minhyuk_clutter_handoff/canonical_three_sphere_config.json
CKPT=flow_deployment/minhyuk_stage1_handoff/checkpoints/hp100_t128_d3.pt
CFG_SHA=b34b4c1a330da8554985476ddb1d4758d9f0a7db24d694c9f3cb6361d39e9455
CKPT_SHA=cc87b65f27506254509b7f4cbbe4734aacfc9e50640a3756cfb0b1ed456e28ff

python scripts/export_lab_flow_frozen_references.py \
  --config "$CFG" \
  --expected-config-sha256 "$CFG_SHA" \
  --checkpoint "$CKPT" \
  --expected-checkpoint-sha256 "$CKPT_SHA" \
  --sampling-temperature 1 \
  --gammas 0.1 0.3 0.5 1.0 \
  --seeds 91000 \
  --device cpu \
  --title "Stage 2 canonical three-sphere pretrained references" \
  --output outputs/minhyuk_stage2_pretrained_frozen_s91000
```

The exporter writes states, raw controls, governed executed controls, dense
positions, a PNG/PDF overlay, and `manifest.json`.

## 3. Closed-loop software smoke in the unchanged deployment harness

```bash
python scripts/run_lab_flow_deployment.py \
  --config "$CFG" \
  --expected-config-sha256 "$CFG_SHA" \
  --checkpoint "$CKPT" \
  --expected-checkpoint-sha256 "$CKPT_SHA" \
  --sampling-temperature 1 \
  --gamma 0.3 \
  --seed 91000 \
  --device cpu \
  --output outputs/minhyuk_stage2_pretrained_closed_loop_g03_s91000
```

This is an interface smoke, not a flight-safety certificate. Collision and
clearance calculations use all three configured spheres. The visual volume is
built from the known map; this handoff does not add onboard perception.
