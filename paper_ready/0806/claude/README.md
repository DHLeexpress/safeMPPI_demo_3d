# Claude workspace

Create one new directory per analysis:

```text
paper_ready/0806/claude/runs/<run_id>/
```

Read shared inputs and Minhyuk's immutable raw logs without modifying them.
Store metrics, figures, reports, commands, source hashes, and an analysis-local
`SHA256SUMS` only in the new run directory. A correction is a new run ID with
`supersedes` recorded; never overwrite the original analysis.
