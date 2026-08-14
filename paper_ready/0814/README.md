# 0814 multi-sphere bowling route handoff

This bundle freezes every trajectory currently exposed by the interactive
bowling site and the exact artifacts needed to audit or re-run it. The public
view remains backward compatible: the original PRE2 and two S4 views are
preserved, while paper-ready/not-paper-ready PRE2, Expanded, SafeMPPI, and
CFM–MPPI are selectable from the same **View** menu.

## What is frozen

| role | artifact | SHA-256 |
|---|---|---|
| PRE2 policy | `checkpoints/pre2/pretrained.pt` | `76b10a69b6f26d65533d4e617ccbf6fb77a2178ac030244c5a745c11a4d0a3c0` |
| expanded S4 policy | `checkpoints/expanded/checkpoint_004.pt` | `a37f93091bfc88340b7f2bab0d41cb889ff59a2caf00d737e58e57fbfa4cdb52` |
| resolved task/scene config | `config/task_config_resolved.json` | `baf3a8f3398cba147696ee783957b865968186e905279d1300a87477459792fc` |
| selected full arrays | `trajectories/paper_ready_bowling_handoff.pt` | see `SHA256SUMS` |
| original legacy site rows | `trajectories/legacy_pre2_s4_site_rows.json` | see `SHA256SUMS` |
| exact rendered site | `site/visualization.html` | see `SHA256SUMS` |

`SITE_TRAJECTORY_INDEX.csv` is the authoritative lookup from View/group to
method, regime, gamma, episode, rollout seed, terminal status, route, source
artifact, and row. The same seed is displayed in the site's focus selector and
trajectory detail card.

The site payload includes 298 view rows. Some are deliberate alternate views
of the same frozen trajectory (for example `legacy-s4` and
`legacy-s4-distinct`), so this is not a count of unique simulator rollouts.

## Paper-ready selection contracts

- PRE2 and Expanded use faithful raw NFE16, M1, temperature 1.0, E15,
  wall250, axis5, control0.05, and obstacle-conditioned speed400. No verifier
  or progress label participates in deployment selection.
- Expanded gamma 0.1 is promoted because its matched search first exposed all
  eight L/R/L route codes in 35 trials. PRE2 is evaluated on the same trial
  budget and exposes three modes.
- SafeMPPI uses the already-published exact source at
  [`paper_ready/0808/safemppi`](../0808/safemppi/README.md) and the pinned
  provenance in [`paper_ready/common/PROVENANCE.md`](../common/PROVENANCE.md).
  This bundle stores only the selected bowling arrays and seed map; it does not
  fork that controller.
- CFM–MPPI is fixed to gamma 0.1 so its paper-ready comparison exactly matches
  Expanded. The former gamma 0.3 paper bank is preserved under
  `not-paper-ready-cfmmppi` together with gamma 0.5 and 1.0. The replacement is
  a fresh, matched 8-seed bank for all three regimes; it is not a relabeling of
  the older trajectories.

## CFM–MPPI contract

CFM–MPPI is PRE2-based, NFE16, with 32 guided proposals, top-8 native-cost
elites, and 32 Gaussian copies per elite. All three ranking stages use the
same native multi-sphere soft cost. H_P/NVP verification and progress labels
are excluded from selection. The fixed regimes are:

| regime | normalized reward | normalized safety |
|---|---:|---:|
| safety | 0.0 | 1.0 |
| performance | 1.0 | 0.0 |
| balanced | 0.5 | 1.0 |

At fixed gamma 0.1, safety and balanced each achieve 5/8 successes and 3/8
collisions, with no OOB or timeout. Their successful mean clearances are
0.106 m and 0.126 m, respectively. Performance is the expected
reward-dominant boundary: 0/8 successes and 8/8 collisions. The eight exact
rollout seeds are shared across all three regimes and recorded in
`trajectories/cfmmppi/site_bank_manifest.json`.

The complete definition, including CBF signed clearance, coefficient scales,
MPPI sigma/lambda, warm start, and calibration boundary, is in
`CFM_MPPI_METHOD_CONTRACT.md`. Exact implementation is in
`source/lab_clutter_cfm_mppi.py` and
`source/run_multisphere_cfm_mppi_bowling.py`; the transitive `safe_mppi`
runtime is frozen under `runtime_snapshot/`.

## Reproduction

First validate the bundle:

```bash
python VERIFY.py
```

Then run on the recorded software stack and matching accelerator family:

```bash
DEVICE=cuda:0 OUT=/tmp/0814_reproduction bash REPRODUCE.sh
```

`REPRODUCE.sh` re-runs the selected PRE2/Expanded seeds, the original CFM–MPPI
approval banks, and the fresh matched gamma 0.1 bank. It passes
`--verify-frozen` for the policy rollouts, so any
state/control/dense-step mismatch aborts. Exact GPU floating-point identity is
bound to the recorded runtime/device; cross-backend reruns are scientific
replications but are not promised to be bit-identical.

The already-frozen `.pt`, JSON, HTML, seed map, and their hashes are the
authoritative byte-identical handoff. `source/build_trajectory_index.py`
rebuilds the CSV lookup, while `source/build_paper_ready_bowling_handoff.py`
and `source/paper-ready-bowling-handoff.html` rebuild the selected site payload
from the original raw banks.

## Why PRE2 is included again

The repository already contains older PRE/SafeMPPI handoffs, but their PRE
checkpoint hash is not this experiment's `76b10a...` PRE2. To avoid an
ambiguous pointer, the exact current PRE2 bytes are included here. SafeMPPI,
whose exact published source is already under `paper_ready/0808`, is referenced
directly rather than duplicated.

## Integrity and provenance

- `SHA256SUMS` binds every file except itself.
- `bundle_manifest.json` records model/source/site identities and the public
  site project.
- `environment.json` records runtime and device contracts.
- `VERIFY.py` checks checksums, required seed metadata, CFM source-to-handoff
  tensor identity, and site seed exposure.
