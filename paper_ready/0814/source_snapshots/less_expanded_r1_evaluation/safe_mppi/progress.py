"""Dependency-free one-line progress display for long research runs."""
from __future__ import annotations


def show_progress(label: str, current: int, total: int, width: int = 24) -> None:
    if total < 1:
        raise ValueError("progress total must be positive")
    current = min(max(int(current), 0), int(total))
    filled = round(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    ending = "\n" if current == total else ""
    print(
        f"\r[{label}] {bar} {current}/{total}",
        end=ending,
        flush=True,
    )


def format_round_summary(row) -> str:
    """One-line per-round expansion summary shared by live and final output.

    Both expansion entry points print this exact text: live as each round
    commits, and again in the end-of-run recap, so a streamed log and a
    post-hoc log carry identical round records.
    """
    return (
        f"  round {row['round']:2d}: verified "
        f"{row['verifier_positives']:3d}/{row['verifier_queries']:3d} -> stored "
        f"{row['positives']:3d}/{row['queries']:3d} "
        f"success {row['success']}/{row['attempted_episode_count']} "
        f"NVP {row['NVP']:3d} "
        f"accepted {row['replay_accepted_positives']:3d} "
        f"replay {row['replay_positives']}/{row['replay_positive_total']} "
        f"GP {row['gp_buffer']:4d} "
        f"(anchor/adapt/evidence {row['gp_anchor_count']}/"
        f"{row['gp_adaptive_count']}/{row['gp_evidence_count']}) "
        f"mESS {row['marginal_ESS_over_K']:.3f} "
        f"sigma {row['sigma_pool_mean']:.3f}->{row['sigma_selected_mean']:.3f} "
        f"uplift {row['uncertainty_uplift']:+.3f} "
        f"steps {row['steps']} time {row['round_total_s']:.1f}s "
        f"loss {row['positive_loss']}"
    )
