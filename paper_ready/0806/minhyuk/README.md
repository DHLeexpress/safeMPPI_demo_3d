# Minhyuk handoff: 0806 flight operator

Work only in a new directory:

```text
paper_ready/0806/minhyuk/runs/<run_id>/
```

The approved suite is
[`../inputs/0806_flight_demonstration_suite/`](../inputs/0806_flight_demonstration_suite/).
It contains two final scenes and 16 frozen 100 Hz references. The old A/B/C and
three-random-scene plans are obsolete.

## Do

1. Check out the pushed `main`, record `git rev-parse HEAD` and preserve dirty
   status if any.
2. Verify both `../LOCKED_SHARED.sha256` and the suite's `FROZEN.sha256` before
   and after the session.
3. Copy `../RUN_LOG_TEMPLATE.md`, `../FLIGHT_INDEX_TEMPLATE.csv`, and
   `../SCENARIO_REGISTRY_TEMPLATE.csv` into the new run directory. Edit only
   those copies.
4. Measure the physical cylinder centers with Vicon and save a run-local
   as-built config. Preserve planned and measured geometry together.
5. Identify the actual Crazyflie hardware runner path, repository SHA and
   exact command. It is not included in this repository.
6. Play the selected `flight_references/*_100hz.npz` at 100 Hz using
   `position_ref`, `velocity_ref` and `acceleration_ref`; log the exact file
   SHA sent and measured Vicon state. `templates/frozen_reference_player.py`
   provides the validated scheduler/callback boundary; copy it into the run
   and connect the callback to the actual runner.
7. Preserve every attempt. Equipment aborts, collisions and unsuccessful
   policies are outcomes, not files to delete.
8. Hash raw telemetry before making figures or videos. Record Drive uploads
   and their hashes.
9. Calibrate the real-log global view against both supplied layouts with
   `../tools/calibrate_global_camera.py`. Save camera JSON and screenshot in
   the run. Capture one fixed physical-camera frame and fill a run-local copy
   of `templates/camera_correspondences.csv` with at least six visible Vicon
   world points before adjusting the interactive virtual camera.
10. Copy `templates/render_real_log.py` into the run before adapting it to the
    actual log. Use `../tools/compose_sim_real.py` to make a derived reviewer
    video with frozen simulation on the left and measured flight on the right.

The observed 2026-08-02 implementation used Vicon measurements and 100 Hz
`cmdFullState(position, velocity, acceleration, yaw, omega)`. Today's frozen
playback differs in one crucial way: it must stream the saved reference rather
than replanning online.

## Do not

- Do not edit, rename, regenerate or overwrite any suite file, exact renderer,
  source NPZ/PT, MP4, scene config or flight reference.
- Do not modify `deploy_sim/` for this campaign.
- Do not rerun SafeMPPI/the policy or resample a trajectory during playback.
- Do not apply ReferenceGovernor, acceleration smoothing or interpolation a
  second time; the reference already contains the governed 100 Hz trajectory.
- Do not overwrite an existing run. Use a new ID and `supersedes`.
- Do not put raw or derived outputs outside `minhyuk/runs/<run_id>/`.
- Do not relabel the outer pretrained gamma-1 collision or inner pretrained
  gamma-1 OOB simulation as a success.
- Do not describe these two hand-designed layouts as an unbiased SR sample.

The immutable simulator video stays unchanged. Any camera-matched real video
or sim/real composite is a derived run artifact and must cite both input SHA
values.
