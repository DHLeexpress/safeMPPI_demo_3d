# SafeMPPI symmetry-breaking supplement

This supplement adds four SafeMPPI references to the 0808 single-sphere
campaign. It uses the exact SafeMPPI implementation from the 0806 campaign:

```text
source Git SHA  9cafc00551e4964b9dbe559b1a4ba95104e9c88a
config SHA      7508a7a76754270e6ffceae8ed9ba3946b5204f5a90d8542c78b55ce835444c2
```

`screen_safemppi_modes.py` refuses to run unless the active `safe_mppi/` tree
is identical to that source commit. The recorder is the unchanged 0806
recorder; every captured rollout was rerun with the vanilla controller and
was bit-identical in states, dense positions, and online nominal polytopes.

## What the finite-seed screen showed

The fixed bank is seeds `0..63`, independently applied at every gamma. These
counts describe SafeMPPI's finite-sample route support; they are not an
unbiased policy evaluation or an expansion result.

| gamma | success | below | above | left | right |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 64/64 | 17 | 20 | 19 | 8 |
| 0.3 | 21/64 | 9 | 9 | 3 | 0 |
| 0.5 | 7/64 | 4 | 3 | 0 | 0 |
| 1.0 | 1/64 | 1 | 0 | 0 | 0 |

The symmetric problem can have multiple equal-cost optima, but a single MPPI
episode must break that symmetry. Finite proposal noise creates small cost
differences; exponential weighting amplifies them, and receding-horizon warm
starts reinforce the selected side. The screen shows the distinction clearly:

- at gamma 0.1, different seeds cover all four route classes;
- as gamma increases, successful support narrows sharply;
- at gamma 1.0, the only success in this bank is below.

This is the comparison point for Reserve G: Safe Flow Expansion packages all
four successful route classes at every gamma, instead of relying on a
particular finite MPPI seed to preserve them.

![SafeMPPI support and representatives](figures/safemppi_support_and_representatives.png)

## Four frozen representatives

These are qualitative representatives, one per gamma. They were selected from
the disclosed bank to cover the three empirically prominent classes. Within
each assigned gamma/mode cell, the selected seed is nearest the cell's median
first-crossing angle. Right is not included because it is the least-supported
pooled class and appears only at gamma 0.1.

| gamma | mode | seed | crossing angle | clearance | time-to-goal |
|---:|---|---:|---:|---:|---:|
| 0.1 | left | 41 | -33.42 deg | 0.15384 m | 8.1 s |
| 0.3 | above | 52 | 103.03 deg | 0.06063 m | 7.8 s |
| 0.5 | below | 12 | -68.80 deg | 0.04555 m | 7.9 s |
| 1.0 | below | 48 | -105.85 deg | 0.01746 m | 7.7 s |

`selection.json` is authoritative. `recordings/` contains the source rollout,
all 512 candidate windows at every replanning step, nominal safety verdicts,
and the actual online nominal polytope history. `flight_references/` contains
the governed 100 Hz references. No planner, governor, smoothing, clipping, or
interpolation is run by the exporter. The committed screen and recordings were
generated on the Mac MPS runtime recorded in `environment.json`.

## Reproduce

Choose a new output directory and a device supported by PyTorch:

```bash
cd paper_ready/0808/safemppi
bash REPRODUCE.sh /tmp/p0808_safemppi_reproduction mps
```

The screen and recorder are deterministic under the same runtime/device, but
the committed recordings remain the authoritative arrays and hashes.
