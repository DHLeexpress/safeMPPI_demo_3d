# 2026-08-06 paper-ready flight plan

Status: **PREPARING_INPUTS**. Dohyun will add and approve the concrete scenes
and trajectories before flight. Minhyuk and Claude must not change shared
files or pre-generated trajectories.

Previous hardware evidence is stored in the
[Drive campaign](https://drive.google.com/drive/u/1/folders/1EjEM4SaClhyBJ4mKbveiexoWvCKFV8vn).
It is context, not an authenticated source snapshot, because its flight
metadata omitted the repository Git SHA.

## Objective

Create three randomly generated five-cylinder scenarios, then fly both:

1. frozen SafeMPPI reference trajectories; and
2. the cylinder-ID pretrained generative policy.

The planned safety levels are `gamma = 0.1, 0.3, 0.5, 1.0`. If all cells are
kept, the matrix is `3 scenarios x 2 policies x 4 gammas = 24 flights`.
Dohyun may remove a cell only before freezing the inputs; the final matrix must
be explicit in `inputs/scenario_registry.csv`.

This date does **not** include a sphere experiment. Both the multiple-sphere
and single-sphere paper configs remain TBD.

## Frozen common contract

- task space: `x=[-2.5,1.3]`, `y=[-1.7,1.8]`, `z=[0.4,2.0]` m
- start: `(-2.1, 1.5, 0.9)` m, zero velocity
- goal: `(0.7, -1.5, 0.9)` m; reach radius `0.2 m`
- five vertical cylinders, sampled from the same geometry law as the 4--8
  cylinder pretrained domain
- physical cylinder radius `0.10 m`; robot inflation `0.10 m`; modeled radius
  `0.20 m`
- no extra obstacle/wall gap; geometry-only admission
- SafeMPPI and policy contracts: [`../common/`](../common/)
- source snapshot: `dabb5011dfc674864e1de275a1e1c2adab58f4af`
- pretrained checkpoint SHA-256:
  `cc87b65f27506254509b7f4cbbe4734aacfc9e50640a3756cfb0b1ed456e28ff`

`lab_pillars_asbuilt.json` is excluded from scene generation. After physically
placing each scenario, the measured Vicon geometry is a new per-run artifact
and must be hashed in the run log.

## Reproduction entry points during preparation

Generate three deterministic five-cylinder candidate episodes with explicitly
chosen seeds. This is a preparation command; its output is not approved until
Dohyun selects the concrete scenes and copies them into `inputs/`.

```bash
python scripts/collect_path_focused_success_quota.py \
  --config paper_ready/common/configs/cylinder_five_generator.json \
  --output outputs/0806_candidate_scenes \
  --fixed-scenes 3 \
  --gammas 0.1 0.3 0.5 1.0 \
  --domain-seed <freeze-me> \
  --rollout-seed-start <freeze-me> \
  --device cuda
```

For each resulting concrete scene, the existing entry points are:

```bash
# Native SafeMPPI offline check for one fixed gamma/seed.
python deploy_sim/run_offline.py \
  --config paper_ready/0806/inputs/scenario_01/concrete_config.json \
  --gamma 0.1 --seed <frozen-seed> --device cpu --gif

# Frozen pretrained-policy references for all gamma values.
python scripts/export_lab_flow_frozen_references.py \
  --config paper_ready/0806/inputs/scenario_01/concrete_config.json \
  --expected-config-sha256 <frozen-config-sha256> \
  --checkpoint flow_deployment/minhyuk_stage1_handoff/checkpoints/hp100_t128_d3.pt \
  --expected-checkpoint-sha256 cc87b65f27506254509b7f4cbbe4734aacfc9e50640a3756cfb0b1ed456e28ff \
  --sampling-temperature 1 \
  --gammas 0.1 0.3 0.5 1.0 \
  --seeds <frozen-seed> \
  --device cpu \
  --output outputs/0806_scenario_01_pretrained
```

Do not pass the randomized template directly to `run.py`: its obstacle arrays
are empty until the path-focused generator materializes a concrete scene.

## Preparation boundary

Dohyun places the final material only in [`inputs/`](inputs/):

- three concrete scene JSON files with exact cylinder centers;
- selected SafeMPPI trajectories and seeds;
- selected pretrained-policy trajectories and seeds;
- `scenario_registry.csv` copied from the template;
- `SHA256SUMS` covering every input.

The randomized template cannot be passed directly to `python run.py` because
its obstacle arrays are empty; use the path-focused generator, then freeze the
resulting concrete config. Likewise, the deployment scripts require a
concrete config.

## Operator boundary

- Minhyuk adds flight outputs only under
  [`minhyuk/runs/<run_id>/`](minhyuk/).
- Claude adds analyses only under
  [`claude/runs/<run_id>/`](claude/).
- Never overwrite an existing run. Corrections receive a new `run_id` and an
  explicit `supersedes` field.
- Copy [`RUN_LOG_TEMPLATE.md`](RUN_LOG_TEMPLATE.md) into each run and complete
  it. Hash the raw log before producing derived figures.
- Minhyuk copies [`FLIGHT_INDEX_TEMPLATE.csv`](FLIGHT_INDEX_TEMPLATE.csv) into
  his workspace and updates only that operator-owned copy.

Before and after a run, verify the frozen shared plan:

```bash
cd paper_ready/0806
shasum -a 256 -c LOCKED_SHARED.sha256
```

After Dohyun freezes `inputs/`, also verify its `SHA256SUMS`. A failed hash is
a stop condition, not permission to regenerate or overwrite a trajectory.

## Required artifacts per flight

- raw Vicon/state telemetry and controller/setpoint log;
- exact trajectory/control archive actually sent;
- source Git SHA and `git status --porcelain` result;
- config, checkpoint, and trajectory SHA-256;
- policy, gamma, sampling temperature, seed, Crazyflie ID, operator, start/end
  timestamp and time zone;
- planned and measured/as-built obstacle geometry;
- terminal status, collision/OOB/timeout flags, clearance and time-to-goal;
- external Drive path and hashes of every uploaded file.

The existing `deploy_sim/` directory is an offline harness, not proof of the
hardware runner used for flight. Record the actual hardware runner path and
commit explicitly.
