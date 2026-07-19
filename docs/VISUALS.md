# Visual atlas

This page indexes every current generated visualization copied into this standalone repository.
The detailed animations came from paired seed-0 runs in the broader 3D experiment; they are provided
for mechanism inspection and are not multi-seed benchmark evidence.

## How to read the three-panel rollout GIFs

Each animation runs at 2 fps and synchronizes:

1. focused XY trajectory with all ten **BLUE online nominal** `H_P` level sets;
2. focused 3D trajectory with all ten **GREEN post-hoc SOCP-verifier** level sets; and
3. the robot-centric encoder field: 80 aligned triangular directions x 9 radial shells = 720 raw
   `H_P` samples.

For horizon `H=10`, the displayed levels use

```text
alpha_h = (1-gamma)^h,  h=1,...,10.
```

Raw `H_P` is geometry-dependent and does not change with gamma at a fixed robot state. Gamma changes
which level is admitted. At gamma 1, all ten thresholds are zero and coincide on the raw boundary.

The BLUE polytope was used online by SafeMPPI. The GREEN variable-face verifier is a post-hoc audit;
it was not inserted into the controller. `validity2` combines the SOCP audit, progress toward the
goal, and task-box containment.

## Construction and gamma-mask views

![80-face uniform triangular construction](assets/uniform_triangular_hp_audit.png)

| in distribution | out of distribution |
|---|---|
| ![ID entry-step gamma mask](assets/gamma_mask_id.gif) | ![OOD entry-step gamma mask](assets/gamma_mask_ood.gif) |

These two GIFs are deterministic geometry probes rather than executed rollouts. They make the
gamma-conditioned entry-step mask visible without pretending that raw `H_P` itself depends on
gamma.

## In distribution: four pillars with an open middle

All seven actual trajectories reached the goal, crossed through the middle, remained collision-free,
and stayed inside the task box.

| gamma | animation | actual outcome | validity2 decomposition |
|---:|---|---|---|
| 0.1 | [GIF](assets/rollouts/actual_id_g0.1_s0_hp10.gif) | success; clearance 0.128 m | FAIL: progress; SOCP PASS |
| 0.2 | [GIF](assets/rollouts/actual_id_g0.2_s0_hp10.gif) | success; clearance 0.115 m | PASS |
| 0.3 | [GIF](assets/rollouts/actual_id_g0.3_s0_hp10.gif) | success; clearance 0.115 m | PASS |
| 0.4 | [GIF](assets/rollouts/actual_id_g0.4_s0_hp10.gif) | success; clearance 0.104 m | PASS |
| 0.5 | [GIF](assets/rollouts/actual_id_g0.5_s0_hp10.gif) | success; clearance 0.073 m | PASS |
| 0.7 | [GIF](assets/rollouts/actual_id_g0.7_s0_hp10.gif) | success; clearance 0.117 m | PASS |
| 1.0 | [GIF](assets/rollouts/actual_id_g1_s0_hp10.gif) | success; clearance 0.072 m | PASS |

![All in-distribution trajectories colored by gamma](assets/pillars_id_gamma_overlay.png)

## Out of distribution: center pillar and grounded overlapping spheres

All seven trajectories remained collision-free and inside the task box. Gamma 0.1 did not reach the
goal; the other six did. The table separates nominal rollout outcome from the stricter post-hoc
`validity2` audit.

| gamma | animation | actual outcome | validity2 decomposition |
|---:|---|---|---|
| 0.1 | [GIF](assets/rollouts/actual_ood_g0.1_s0_hp10.gif) | no goal/no crossing; clearance 0.071 m | FAIL: progress; SOCP PASS |
| 0.2 | [GIF](assets/rollouts/actual_ood_g0.2_s0_hp10.gif) | success; lower detour; clearance 0.146 m | PASS |
| 0.3 | [GIF](assets/rollouts/actual_ood_g0.3_s0_hp10.gif) | success; inside field; clearance 0.036 m | FAIL: progress; SOCP PASS |
| 0.4 | [GIF](assets/rollouts/actual_ood_g0.4_s0_hp10.gif) | success; inside field; clearance 0.031 m | FAIL: progress; SOCP PASS |
| 0.5 | [GIF](assets/rollouts/actual_ood_g0.5_s0_hp10.gif) | success; inside field; clearance 0.016 m | PASS |
| 0.7 | [GIF](assets/rollouts/actual_ood_g0.7_s0_hp10.gif) | success; inside field; clearance 0.046 m | FAIL: progress; SOCP PASS |
| 1.0 | [GIF](assets/rollouts/actual_ood_g1_s0_hp10.gif) | success; inside field; clearance 0.021 m | FAIL: SOCP at k=40,42; progress PASS |

## Important interpretation

An executed trajectory point from a later receding-horizon control period is not required to remain
inside an earlier frozen ten-step nominal polytope. The online guarantee being checked is the first
executed transition against the polytope built at that same control period. Consequently, a frozen
horizon drawn over the later receding-horizon path is a diagnostic, not an online-failure count.
