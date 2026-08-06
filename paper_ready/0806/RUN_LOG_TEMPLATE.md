# Flight run log

Copy this file to `<operator>/runs/<run_id>/RUN_LOG.md` and fill every field.
Use `TBD` only before launch; a completed run may not contain `TBD`.

## Identity

- run ID:
- supersedes: none
- date/time/time zone:
- operator:
- Crazyflie ID:
- scenario ID:
- policy: `SafeMPPI` / `pretrained_flow`
- gamma:
- seed:
- sampling temperature:

## Source integrity

- repository Git SHA:
- `git status --porcelain`: empty / attach output
- command or hardware-runner invocation:
- hardware-runner repository/path/SHA:
- config path and SHA-256:
- checkpoint path and SHA-256, or `N/A` for SafeMPPI:
- frozen trajectory path and SHA-256:
- input manifest verification: PASS / FAIL
- locked shared manifest verification: PASS / FAIL

## Physical setup

- planned obstacle config path and SHA-256:
- measured Vicon/as-built config path and SHA-256:
- frame/calibration source and SHA-256:
- takeoff/start check:
- geofence check:

## Outcome

- terminal status: success / collision / OOB / timeout / aborted
- goal reached:
- minimum physical clearance:
- time-to-goal:
- safety intervention or manual abort:
- anomaly notes:

## Artifacts

- raw telemetry:
- controller/setpoint log:
- sent trajectory/control archive:
- figures/video:
- Drive folder:
- artifact `SHA256SUMS`:
