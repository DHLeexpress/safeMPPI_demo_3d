# Compact r20 expansion result

This directory preserves the pretrained `checkpoint_000.pt` and selected
`checkpoint_020.pt` needed for a matched endpoint evaluation.
It is an **experimental coverage checkpoint**, not a qualified controller:
raw temperature-1 \(M=50/\gamma\) evaluation yielded SR \(0.20\), CR
\(0.505\), OOB \(0.295\), and window validity \(0.838\).

The expanded distribution discovered successful above routes
(\(23/200\)), whereas the pretrained model had none, but it forgot the
pretrained below route and lost task success. See `manifest.json` and
`raw_m50_seed191000.json` for the exact counts.

Quick \(M=20/\gamma\) endpoint check:

```bash
PRE=flow_deployment/minhyuk_handoff/expansion_pretrain
EXP=results/lab_ball_expansion/minhyuk_nozB5_r20

python scripts/evaluate_ball_expansion.py \
  --pretrain-dir "$PRE" \
  --expansion "$EXP" \
  --episodes 20 \
  --probe-samples 16 \
  --stride 20 \
  --seed 91000 \
  --raw-tight-corridor \
  --screening-only
```

Exact disjoint \(M=50/\gamma\), seed-191000 headline audit:

```bash
PRE=flow_deployment/minhyuk_handoff/expansion_pretrain
EXP=results/lab_ball_expansion/minhyuk_nozB5_r20

python scripts/evaluate_ball_expansion.py \
  --pretrain-dir "$PRE" \
  --expansion "$EXP" \
  --episodes 50 \
  --probe-samples 32 \
  --stride 20 \
  --seed 191000 \
  --raw-tight-corridor \
  --screening-only
```

This compact result has no `events.pt`, because the original sweep used
`--event-log none`; therefore it can regenerate raw curves and galleries but
not an acquisition-mechanism video.
