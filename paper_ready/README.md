# Paper-ready flight campaigns

This directory is the provenance boundary for experiments intended for the
paper. Each date has one immutable campaign plan and append-only operator
workspaces.

## Non-negotiable rule

1. Dohyun owns `<date>/inputs/` and freezes the exact concrete scenes and
   trajectories before flight.
2. Once frozen, nobody edits the date plan, shared inputs, or an existing run.
3. Minhyuk and Claude add new work only under their own
   `<date>/<operator>/runs/<run_id>/` directory.
4. A correction is a new run directory. Never overwrite a trajectory, config,
   log, metric, or figure used by a prior flight.
5. Every run records the repository commit, clean/dirty state, command,
   config/checkpoint/trajectory SHA-256 values, gamma, seed, hardware identity,
   Vicon/as-built geometry, outcome, and external Drive location.

Commit compact metadata, metrics, and paper figures here. Large raw telemetry
or video may remain in Drive, but its exact Drive path, byte count, and SHA-256
must be recorded in the run-local manifest.

The shared mathematical and software contracts are pinned in
[`common/`](common/). Frozen campaigns are [`0806/`](0806/) and
[`0808/`](0808/).

## Why the repository SHA is mandatory

The earlier Drive campaign contains useful per-flight NPZ/CSV/meta/figures,
but its `_meta.json` files do not record a source Git SHA. Time proximity can
reconstruct a likely checkout, but cannot authenticate it. Paper-ready runs
must therefore record both the commit and byte hashes of every effective
input.
