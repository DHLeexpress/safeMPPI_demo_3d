# PRE2 mirrored-cylinder gamma 0.3 references

This bundle freezes eight successful trajectories from the exact randomized,
axis-180-mirrored vertical-cylinder demonstrations used to train PRE2.  The
fixed context is `gamma=0.3`.

## What is included

- Four independently randomized scene pairs, each with its source and exact
  axis-180 mirror member.
- Eight trajectories with one of every three-bit lateral signature:
  `LLL`, `LLR`, `LRL`, `LRR`, `RLL`, `RLR`, `RRL`, `RRR`.
- The original 10 Hz demo `.npz` files, a self-contained PyTorch bank, and
  reconstructed 100 Hz position/velocity/acceleration references.
- Exact rollout seed, scene seed/hash, pair ID/member, cylinder geometry,
  source hashes, task config, and PRE2 training manifest.

The signature is the left/right sign of the trajectory's lateral displacement
from the infinite XY start-goal axis at 25%, 50%, and 75% longitudinal
progress.  It is a reproducible path-shape label, not a claim that random
cylinder scenes share one common obstacle homotopy.

## Important scope

These are exact accepted **training demonstrations** from the PRE2 data law,
not fresh policy evaluations on one common scene.  Each focused trajectory in
the public visualization therefore displays its own cylinder scene.  The
references are marked `hardware_eligible=false`; they should only be flown if
the corresponding cylinder scene is physically reconstructed and separately
approved.

## Files

- `manifest.json`: authoritative roster and provenance.
- `FLIGHT_INDEX.csv`: compact index for all eight rows.
- `trajectories/raw_demo_npz/`: byte-identical source demonstrations.
- `trajectories/gamma0p3_mode8_bank.pt`: convenient full-state bank.
- `flight_references/`: 100 Hz position, velocity, and acceleration arrays.
- `site_rows.json`: uncompressed visualization handoff.
- `SHA256SUMS`: hashes for every file in this sub-bundle.

The referenced PRE2 checkpoint is `../checkpoints/pre2/pretrained.pt`, SHA256
`76b10a69b6f26d65533d4e617ccbf6fb77a2178ac030244c5a745c11a4d0a3c0`.

## Verify

From `paper_ready/0814`:

```bash
python mirrored_cylinder_gamma03/VERIFY.py
```

The verifier checks file hashes, the fixed gamma, all eight signatures, four
complete source/mirror pairs, success/collision/OOB metadata, and the 100 Hz
reference recurrence errors recorded in the manifest.
