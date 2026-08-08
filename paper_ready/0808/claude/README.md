# Claude handoff: 0808 flight analysis

Work only in a new directory:

```text
paper_ready/0808/claude/runs/<run_id>/
```

## Do

- Verify `../SHA256SUMS` before reading the campaign.
- Treat the frozen bundle and Minhyuk's raw run directories as read-only.
- Copy analysis or plotting code into the new Claude run before modifying it.
- Join real logs to planned references by `flight_id` and exact SHA, not by
  filename similarity or directory order.
- Preserve simulation failure, hardware abort, collision, and tracking error
  as distinct outcomes.
- Record every input SHA, command, metric, figure, and Drive destination in a
  run-local manifest and `SHA256SUMS`.

## Do not

- Do not edit shared files, frozen references, Minhyuk logs, or an existing
  Claude run.
- Do not rerun or replace Reserve G, change NFE 12, regenerate a mode, or
  silently substitute a later expanded checkpoint.
- Do not modify `deploy_sim/`, re-govern references, or hide either known
  pretrained collision.
- Do not report the curated four-mode selection as an unbiased coverage or SR
  estimate.
- Delete only temporary files created by the current analysis after their
  retained outputs are hashed.
