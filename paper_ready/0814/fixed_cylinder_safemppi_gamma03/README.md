# Fixed-scene SafeMPPI · gamma 0.3

This bundle freezes eight **SafeMPPI controller rollouts** on one fixed,
axis-symmetric vertical-cylinder episode. It is not a PRE2 rollout bank.

## Intended visual story

The same four obstacle boundaries are used for every trajectory:

- `ALL_LEFT` (`LLLL`): 3 references that pass every cylinder boundary on the
  left.
- `ALL_RIGHT` (`RRRR`): 3 references that pass every cylinder boundary on the
  right.
- `MIDDLE`: 2 rare weaving references (`LRLR`, `RRLL`).

In the 96-seed fixed-scene screen, 87 rollouts were nominal-safe successes:
`LLLL=57`, `RRRR=27`, `LRLR=2`, `RRLL=1`. Thus the two outside routes are the
dominant behaviors and middle routes are genuinely rare.

## Files

- `scene.json`: the one fixed start/goal/bounds/cylinder episode.
- `manifest.json`: authoritative seeds, metrics, screening counts, and source
  hashes.
- `raw_rollouts/`: 10 Hz state, raw control, applied control, and exact 100 Hz
  dense-position evidence.
- `flight_references/`: 100 Hz position/velocity/acceleration references.
- `FLIGHT_INDEX.csv`: compact roster for Minhyuk.
- `REPRODUCE.py`: reruns the exact SafeMPPI controller seeds.
- `VERIFY.py`: hashes, fixed-scene identity, route roster, and recurrence checks.
- `SHA256SUMS`: byte-level integrity manifest.

All references are marked `hardware_eligible=false`; hardware execution still
requires operator and lab safety approval.

## Verify

From the repository root:

```bash
python paper_ready/0814/fixed_cylinder_safemppi_gamma03/VERIFY.py
```

## Reproduce controller rollouts

The script imports the already-frozen 0814 SafeMPPI runtime snapshot and checks
its SHA256 before running:

```bash
python paper_ready/0814/fixed_cylinder_safemppi_gamma03/REPRODUCE.py \
  --output /tmp/fixed-cylinder-safemppi-gamma03
```

The eight frozen rollout seeds are:

```text
ALL_LEFT : 819585 819574 819558
ALL_RIGHT: 819510 819552 819520
MIDDLE   : 819546 (LRLR), 819507 (RRLL)
```
