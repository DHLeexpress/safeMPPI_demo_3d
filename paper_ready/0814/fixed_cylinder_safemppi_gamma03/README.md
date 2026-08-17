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

In the 384-seed fixed-scene screen, 333 rollouts were nominal-safe successes:
`LLLLLL=165`, `RRRRRR=81`, `LRLRLR=61`, `RRLRLL=25`, `LRLRLL=1`. Thus the
two outside routes are dominant and the roster retains two distinct middle
signatures. Within each frozen route quota, the final references prioritize
small mean and p90 absolute deviation from `z=0.9`, while retaining at least
`0.012 m` modeled clearance.

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
ALL_LEFT : 820864 820701 820723
ALL_RIGHT: 820829 820870 820764
MIDDLE   : 820518 (LRLRLR), 820627 (RRLRLL)
```
