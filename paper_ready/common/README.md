# Shared experiment contract

These files are immutable snapshots for paper-ready campaigns. They do not
replace the live source files; provenance below points to the authoritative
location and source commit.

| item | snapshot | authoritative source at `dabb5011dfc674864e1de275a1e1c2adab58f4af` |
|---|---|---|
| common task space | [`taskspace.json`](taskspace.json) | cylinder-ID config below |
| SafeMPPI recipe | [`safemppi_spec.json`](safemppi_spec.json) | `safe_mppi/controller.py` plus config |
| pretrained ID distribution | [`configs/cylinder_id_4to8.json`](configs/cylinder_id_4to8.json) | `configs/lab_clutter_cylinders_path_midpoint_uniform_v2.json` |
| 0806 five-cylinder candidate generator | [`configs/cylinder_five_generator.json`](configs/cylinder_five_generator.json) | `configs/lab_clutter_cylinders_lab_five_v2.json` |
| multiple-sphere task | [`configs/multiple_sphere.TBD.md`](configs/multiple_sphere.TBD.md) | not frozen |
| single-sphere task | [`configs/single_sphere.TBD.md`](configs/single_sphere.TBD.md) | not frozen |

Exact source locations, commits, and SHA-256 values are in
[`PROVENANCE.md`](PROVENANCE.md).

Once a dated campaign references a shared JSON or provenance snapshot, that
file is immutable. A later recipe uses a new versioned filename; it does not
rewrite the old snapshot. Index and TBD documents may evolve and are therefore
not part of a historical campaign's hash lock.

Important: `lab_pillars_asbuilt.json` is deliberately not a trajectory
generation input. It describes a Vicon-measured physical arrangement, not the
pretrained model's randomized 4--8-cylinder distribution. A new as-built file
must be recorded per physical scenario after placement.
