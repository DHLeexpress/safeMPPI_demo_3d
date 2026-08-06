# Frozen 0806 flight-demonstration suite

This directory is the immutable input bundle for the 2026-08-06 flight day.
It contains two final hand-designed five-cylinder layouts and eight simulated
trajectories per layout:

| scene | SafeMPPI seed | pretrained seed | config SHA-256 |
|---|---:|---:|---|
| `symmetric_scene_outer` | 0 | 91884 | `549e47bb70542a1bcce345bb8556c1b8d5aede050ddb6afc3acc3ab099e48a02` |
| `symmetric_scene_inner` | 0 | 91015 | `2c22ce1de40096710c9a08f6ba2d20d3cbf1520d0ceabe5d35883c3de5af7c3a` |

Each policy is represented at `gamma = 0.1, 0.3, 0.5, 1.0`, giving exactly
`2 scenes x 2 policies x 4 gammas = 16` planned trajectory cells. These two
layouts replace every earlier A/B/C or three-random-scene proposal.

## Scientific scope

The layouts were designed directly as symmetric `1-2-2` cylinder patterns.
They were **not** sampled by `path_focused_midpoint_uniform_v2`. They are
paper demonstration inputs, not an unbiased success-rate evaluation or a
sample from the training distribution. The source `scene_meta.json` files
record that distinction explicitly.

The simulator source snapshot is
`9cafc00551e4964b9dbe559b1a4ba95104e9c88a`. The pretrained checkpoint is
[`../../../../flow_deployment/minhyuk_stage1_handoff/checkpoints/hp100_t128_d3.pt`](../../../../flow_deployment/minhyuk_stage1_handoff/checkpoints/hp100_t128_d3.pt),
SHA-256
`cc87b65f27506254509b7f4cbbe4734aacfc9e50640a3756cfb0b1ed456e28ff`.

## What is frozen

For every scene and policy, the bundle preserves:

- the exact concrete scene config;
- the 10 Hz state/control archive and 100 Hz governed path (`run_*.npz`);
- the full per-step candidate/polytope event log (`events_run_*.pt`);
- the original simulation MP4 and provenance sidecar;
- an exported 100 Hz flight reference with time, position, velocity and
  acceleration; and
- the exact recorder, renderer and validator source used for the simulation.

`FROZEN.sha256` covers every file except itself. Verify it before and after a
flight session:

```bash
cd paper_ready/0806/inputs/0806_flight_demonstration_suite
shasum -a 256 -c FROZEN.sha256
```

Do not edit or regenerate anything here. New physical logs, calibrated camera
files and derived sim/real videos belong under `paper_ready/0806/minhyuk/runs/`.

## Simulation outcomes are provenance, not a flight filter

| scene | policy | gamma .1 | gamma .3 | gamma .5 | gamma 1.0 |
|---|---|---|---|---|---|
| outer | SafeMPPI | success | success | success | success |
| outer | pretrained | success | success | success | **collision** |
| inner | SafeMPPI | success | success | success | success |
| inner | pretrained | success | success | success | **OOB** |

All 16 references are preserved because the experiment studies both successful
and failed behavior. A simulated `success` is not a hardware-safety guarantee,
and the two explicitly failed policy references must not be relabeled.

## Reference playback contract

The source rollout already applied the repository `ReferenceGovernor` exactly
once. Every file under `flight_references/` contains:

- `time_s`, `position_ref`, `velocity_ref`, `acceleration_ref` at 100 Hz;
- the original 10 Hz raw and governed acceleration controls;
- gamma, seed and control/reference rates.

During frozen playback, do not call the policy or SafeMPPI, do not reapply the
governor, and do not add interpolation or smoothing. The hardware runner sends
the stored reference and records measured Vicon state separately. The operator
must record the actual runner path and Git SHA because the hardware runner is
not part of this frozen simulation bundle.

## Byte-identical simulation video reproduction

The exact sources are in [`renderer/`](renderer/). The verified environment is
specified in [`rendering_environment.json`](renderer/rendering_environment.json).
The private 591 MB Python runtime is deliberately not committed. In a matching
Helios environment:

```bash
python paper_ready/0806/inputs/0806_flight_demonstration_suite/renderer/reproduce_sim_videos.py \
  --repo "$PWD" \
  --suite paper_ready/0806/inputs/0806_flight_demonstration_suite \
  --output /tmp/p0806_reproduced_videos
```

The command re-renders all 16 MP4s from the checked-in NPZ/PT inputs and fails
unless every video SHA-256 is byte-identical to the frozen original. Never edit
the exact recorder/renderer sources. Copy them into an operator-owned run only
when building a new visualization from measured data.

The completed 16/16 result is preserved in
[`BYTE_IDENTICAL_REPRODUCTION.json`](renderer/BYTE_IDENTICAL_REPRODUCTION.json).

## Camera matching and reviewer video

The frozen simulation's global camera is elevation `25 deg`, azimuth `-57 deg`,
world-up `+z`, view angle `30 deg`; the second panel is the unchanged
velocity-aligned ego camera. Minhyuk first calibrates the real-log rendering
camera against both layouts with:

```bash
python paper_ready/0806/tools/calibrate_global_camera.py \
  --outer-config paper_ready/0806/inputs/0806_flight_demonstration_suite/scenes/symmetric_scene_outer/concrete_config.json \
  --inner-config paper_ready/0806/inputs/0806_flight_demonstration_suite/scenes/symmetric_scene_inner/concrete_config.json \
  --output paper_ready/0806/minhyuk/runs/<run_id>/camera
```

Press `s` to save the camera JSON and screenshot. The calibrated real-log video
is rendered from a run-local copy of
[`render_real_log.py`](../../minhyuk/templates/render_real_log.py), then paired
with the untouched simulation MP4 using
[`compose_sim_real.py`](../../tools/compose_sim_real.py). The composite is a
derived artifact; the left simulation source remains byte-identical.
