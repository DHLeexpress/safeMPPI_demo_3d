# Frozen Flow Policy → `deploy_sim`

For the current lab-native pretrained checkpoint, runtime sampling temperature,
online state-feedback runner, and frozen-reference exporter, use
[`minhyuk_handoff/`](minhyuk_handoff/). That handoff does not use the temporary
canonical-to-lab frame bridge documented below.

This folder is a deliberately temporary interconnection layer. It loads one
already-trained B1 ball-flow checkpoint and calls it inside Minhyuk's unchanged
offline Crazyflie loop. It does **not** collect motion-capture data, expand a
policy online, add a safety guarantee, or modify `deploy_sim`.

## Contract

The canonical training task is

\[
s_c=(0,0,2),\quad g_c=(3,0,2),\quad
o_c=(1.5,0,2),\quad r_c=0.254.
\]

The lab task comes directly from
`configs/crazyflie_mppi_corner.json`. The temporary map is

\[
p_l=s_l+aR(p_c-s_c),\qquad
a=\frac{\lVert g_l-s_l\rVert}{\lVert g_c-s_c\rVert},
\]

where \(R\) maps the source forward/left/up basis to the corresponding lab
basis. Thus start and goal match exactly without reflection. The real lab
sphere is transformed back into the policy frame and used to build the 10-D
context

\[
c=[g-p,\;v,\;b_{\rm near}-p,\;\gamma].
\]

The sphere does not exactly coincide with the training sphere after this
endpoint fit; that residual is reported as OOD rather than hidden.

The policy samples one \(H=10\), 3-D acceleration plan at raw temperature 1 and
executes its first action. Direction is rotated into the lab frame. Magnitude is
scaled by the controller-authority ratio \(0.3/1.0\), not by geometric
similarity, and uniformly limited to the lab configuration. Keeping the 0.1 s
replan period while changing both spatial scale and acceleration authority is
not a dynamically exact double-integrator similarity. The test is therefore a
software interconnection diagnostic, not a flight certificate.

Also note that Minhyuk's unchanged harness passes its governed reference
velocity \(v_{\rm ref}\), rather than its internally estimated measured
velocity, as the second half of the controller state. The trace preserves that
quantity so the interface difference is inspectable.

## Reproduce

```bash
python scripts/run_flow_deployment.py \
  --config configs/crazyflie_mppi_corner.json \
  --pretrain-dir results/global50_reference/pretrain_global10_h48p32_s0 \
  --episodes 20 \
  --seed-start 0 \
  --output outputs/flow_deployment/pretrained_corner \
  --gif
```

To test a frozen expanded checkpoint, add
`--expansion PATH_TO_RUN --round N`. The runner writes:

- per-episode metrics and a machine-readable run contract;
- the complete controller trace, including canonical contexts and both plans;
- Minhyuk's native CSV/NPZ/polytope visualization;
- PNG/PDF frame-comparison figures.

`deploy_sim_lock.json` pins every Minhyuk deployment file used by this test.
The runner checks those hashes both before and after execution and refuses to
run if any pinned file differs.
