# Dohyun-owned frozen inputs

Only Dohyun prepares this directory. Before flight it must contain:

```text
scenario_registry.csv
scenario_01/
  concrete_config.json
  safemppi/
  pretrained/
scenario_02/
  ...
scenario_03/
  ...
SHA256SUMS
```

Each policy directory stores the exact trajectory/control files, seeds, and a
manifest for all approved gamma values. After `SHA256SUMS` is committed, this
directory is immutable. Minhyuk and Claude consume it read-only and put all
new logs in their own `runs/<run_id>/` directories.
