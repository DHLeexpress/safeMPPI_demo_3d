# Minhyuk handoff: 0808 single-sphere flight operator

Work only in a new directory:

```text
paper_ready/0808/minhyuk/runs/<run_id>/
```

The authoritative operator index is
[`../FLIGHT_INDEX_ALL.csv`](../FLIGHT_INDEX_ALL.csv).
It contains 30 simulated successes and two known pretrained collisions. The
collision references are simulation-only negative controls and must not be
sent to hardware unchanged.

## Do

1. Check out the pinned `main`, record `git rev-parse HEAD`, and run
   `python source/bundle_integrity.py` before and after the session.
2. Copy [`../RUN_LOG_TEMPLATE.md`](../RUN_LOG_TEMPLATE.md) and
   [`templates/frozen_reference_player.py`](templates/frozen_reference_player.py)
   into a new run directory. Edit only those copies.
3. Select a reference by its exact `flight_id`, then record both its NPZ SHA
   and the source rollout SHA from the index. Never select by glob order.
4. Validate the reference locally with `--expected-sha256` before connecting
   the `send_full_state(position, velocity, acceleration)` callback to the
   actual Crazyflie runner.
5. Stream the stored reference once at 100 Hz. Record command timestamps,
   dropped/deadline-missed packets, Vicon state, Crazyflie ID, operator,
   aborts, and the exact hardware-runner path/SHA/command.
6. Preserve every attempt and hash raw telemetry before producing derived
   figures or uploading to Drive.

## Expanded Reserve G: extra care

- `expanded_v1_reserve_G_nfe12.pt` is the only expanded checkpoint in this
  campaign. It was sampled with packaged **NFE 12**, not the pretrained
  checkpoint's NFE 16.
- The original 16 modes plus the six-trajectory angular supplement are
  **22 frozen seed trajectories**, not a discrete mode input
  accepted by the network. To fly a named `below/above/left/right` mode, use
  the corresponding frozen 100 Hz reference. Re-inference does not guarantee
  that mode.
- The checkpoint and GPU-specific reproduction code are provenance assets,
  not dependencies of normal flight playback. Do not load the expanded model
  on the flight computer for this campaign.
- `ReferenceGovernor` has already been applied exactly once. The 100 Hz files
  already contain governed position, velocity, and acceleration.
- A future expanded checkpoint receives a new versioned campaign artifact;
  never replace Reserve G in place or alias it to a generic `expanded.pt`.

### Expanded angular supplement

- Gamma 1.0 now has exactly one frozen reference in each 45-degree crossing
  section S0--S7. The complete seed/angle table is in `../README.md` and the
  corresponding flight files are indexed as group `expanded_supplement_v1`
  together with the four original gamma-1 references.
- Gamma 0.1 adds left-side-above seed `92851` and right-side-above seed
  `91555`. Prefer these two over straight-above seed `108992` when an overhead
  suspension line makes the direct vertical route undesirable.
- These are already generated and governed 100 Hz references. Do not run the
  checkpoint, resample their seeds, or interpolate them during playback.
- Every new reference remains subject to operator and hardware safety
  approval; the geometric side label does not itself certify the real flight.

## SafeMPPI supplement

- The four `safemppi_prominent_modes` references use the exact 0806 SafeMPPI
  source tree at `9cafc00551e4964b9dbe559b1a4ba95104e9c88a`.
- They are frozen 100 Hz references like the policy trajectories. Do not run
  SafeMPPI online, resample a seed, or rebuild a nominal polytope during
  playback.
- The four paths are qualitative prominent-class representatives, not an
  unbiased success-rate result. Their full finite-seed context is in
  [`../safemppi/README.md`](../safemppi/README.md).
- Candidate histories and the actual online nominal-polytopes are provenance
  assets under `../safemppi/recordings/`; hardware playback needs only the
  indexed 100 Hz reference.

## Do not

- Do not edit or overwrite checkpoints, selections, source trajectories,
  frozen references, figures, runtime snapshots, manifests, or checksums.
- Do not modify `deploy_sim/` for this campaign.
- Do not rerun a policy or SafeMPPI, change gamma/seed/NFE/temperature, or
  resample a trajectory during playback.
- Do not apply a governor, smoothing, clipping, or interpolation a second
  time.
- Do not send either `SIMULATION_ONLY_KNOWN_COLLISION` reference to hardware.
- Do not overwrite an existing run or delete failed/aborted attempts.
- Do not claim the curated 16-mode bundle is an unbiased SR evaluation.

The actual Crazyflie hardware runner remains Minhyuk-owned and is not bundled
here. The supplied player is validation-only until its callback is connected
inside a run-local copy.
