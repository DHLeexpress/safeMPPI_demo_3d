# Fixed-scene SafeMPPI · gamma 0.3

This bundle freezes eight **SafeMPPI controller rollouts** on one fixed,
axis-symmetric six-cylinder episode. It is not a PRE2 rollout bank.

## Intended visual story

The same six obstacle boundaries are used for every trajectory:

- `ALL_LEFT` (`LLLLLL`): 3 references that pass every cylinder boundary on the
  left.
- `ALL_RIGHT` (`RRRRRR`): 3 references that pass every cylinder boundary on the
  right.
- `MIDDLE`: 2 weaving references (`LRLRLR`, `RRLRLL`).

In the 96-seed fixed-scene screen, 87 rollouts were nominal-safe successes:
`LLLLLL=37`, `RRRRRR=24`, `LRLRLR=23`, `RRLRLL=3`. Thus the two outside
routes are dominant and the roster retains two distinct middle signatures.

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
ALL_LEFT : 820565 820534 820578
ALL_RIGHT: 820586 820514 820560
MIDDLE   : 820577 (LRLRLR), 820544 (RRLRLL)
```
