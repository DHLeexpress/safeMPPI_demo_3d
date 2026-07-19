"""Placeholder only: safe expansion is intentionally not implemented in this package yet."""


def run_safe_expansion(dataset_dir, output_dir, **kwargs):
    """Future entry point consuming the per-gamma NPZ rollouts written by ``acquire.py``."""
    raise NotImplementedError(
        "Safe expansion is the next stage. Freeze its acceptance rule, model, and training recipe "
        "before replacing this placeholder; do not infer them from the acquisition code.")
