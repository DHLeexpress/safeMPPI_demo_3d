# Flight run log

Copy this file to `<operator>/runs/<run_id>/RUN_LOG.md` and fill every field.
Use `TBD` only before launch; a completed run may not contain `TBD`.

## Identity

- run ID:
- supersedes: none
- date/time/time zone:
- operator:
- Crazyflie ID:
- scene: `symmetric_scene_outer` / `symmetric_scene_inner`
- policy: `safemppi` / `pretrained`
- gamma:
- seed:
- sampling temperature: `N/A` for SafeMPPI / `1.0` for pretrained
- simulated outcome recorded in frozen suite:

## Source integrity

- repository Git SHA:
- `git status --porcelain`: empty / attach output
- actual hardware-runner repository/path/SHA:
- exact hardware-runner invocation:
- config path and SHA-256:
- checkpoint path and SHA-256, or `N/A` for SafeMPPI:
- frozen 100 Hz reference path and SHA-256:
- source rollout NPZ and SHA-256:
- source simulation MP4 and SHA-256:
- frozen suite manifest verification: PASS / FAIL
- locked shared manifest verification: PASS / FAIL

## Physical setup and camera

- planned obstacle config path and SHA-256:
- measured Vicon/as-built config path and SHA-256:
- Vicon/world-frame calibration path and SHA-256:
- visualization camera JSON and SHA-256:
- physical-camera frame or 3-D-to-2-D correspondence evidence:
- takeoff/start check:
- geofence check:

## Playback contract

- stream rate:
- governor/smoothing applied by playback: **must be no**
- policy/planner called during playback: **must be no**
- first/last sent reference timestamps:
- exact sent setpoint/controller log:

## Outcome

- terminal status: success / collision / OOB / timeout / equipment abort
- goal reached:
- minimum physical clearance:
- time-to-goal:
- Vicon dropout or pose glitch:
- safety intervention or manual abort:
- anomaly notes:

## Artifacts

- raw telemetry:
- controller/setpoint log:
- sent trajectory/control archive:
- real-log video:
- sim/real composite:
- figures:
- Drive folder:
- artifact `SHA256SUMS`:
