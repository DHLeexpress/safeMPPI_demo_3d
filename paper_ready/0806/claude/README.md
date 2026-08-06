# Claude handoff: 0806 analysis

Work only in a new directory:

```text
paper_ready/0806/claude/runs/<run_id>/
```

## Do

- Verify `../LOCKED_SHARED.sha256` and
  `../inputs/0806_flight_demonstration_suite/FROZEN.sha256`.
- Read the frozen suite and Minhyuk's raw run directories without modifying
  either.
- Copy source or visualization templates into the new Claude run before
  changing them.
- Preserve equipment failures and scientific failures separately; keep every
  attempt in the denominator appropriate to its declared analysis.
- Cite the exact scene, trajectory, real-log, camera-calibration, source and
  checkpoint hashes for every metric or figure.
- Build sim/real videos only as new derived files and verify that the source
  simulation MP4 SHA is unchanged.
- Store commands, metrics, figures, reports and a run-local `SHA256SUMS` in the
  analysis directory. Corrections use a new run ID with `supersedes`.

## Do not

- Do not edit shared files, the frozen suite, Minhyuk's raw logs or an existing
  Claude run.
- Do not regenerate trajectories, change seeds/gamma, rerun the controller or
  replace the exact renderer inside the suite.
- Do not modify `deploy_sim/`.
- Do not use the obsolete A/B/C or random-three-scene plan.
- Do not call these two hand-designed demonstration layouts an unbiased
  evaluation set, and do not hide collision/OOB/equipment-abort outcomes.
- Do not delete `/tmp` material belonging to another process. Delete only
  temporary files created by your own run after their outputs are hashed.
