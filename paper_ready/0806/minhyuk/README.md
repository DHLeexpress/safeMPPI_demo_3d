# Minhyuk workspace

Create one new directory per physical run:

```text
paper_ready/0806/minhyuk/runs/<run_id>/
```

Copy `../../RUN_LOG_TEMPLATE.md` into it, save raw telemetry before derived
artifacts, and generate a run-local `SHA256SUMS`. Read `../../inputs/` only;
never edit or regenerate a frozen trajectory in place. A retry is a new run ID.

At campaign start, copy `../../FLIGHT_INDEX_TEMPLATE.csv` to
`flight_index.csv` in this directory. Fill that operator-owned copy as flights
finish; do not edit the shared template.
