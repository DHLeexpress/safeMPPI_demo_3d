"""Pretrain and qualify the stateless Minhyuk-frame reference flow policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.expansion import mean_pairwise_lengthscale
from safe_mppi.flow_model import ConditionalFlowMLP
from safe_mppi.lab_reference_flow_task import (
    LAB_RAW_CONTEXT_SCHEMA,
    LAB_REFERENCE_CONTEXT_DIM,
    LAB_ROUTE_MODES,
    lab_reference_demo_windows,
    raw_reference_rollout,
)
from safe_mppi.lab_visual_flow import (
    LAB_VISUAL_CHANNELS,
    LAB_VISUAL_FRAME,
    LAB_VISUAL_GRID_SHAPE,
    LAB_VISUAL_PACKED_DIM,
    LAB_VISUAL_SCHEMA,
    LabVisualFlowPolicy,
)


ROOT = Path(__file__).resolve().parents[1]


def trajectory_split(metadata: list[dict], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Disjoint train/validation split over whole demonstration trajectories."""
    groups = sorted({(float(row["gamma"]), int(row["seed"])) for row in metadata})
    generator = np.random.default_rng(seed)
    validation_groups = set()
    for gamma in sorted({group[0] for group in groups}):
        candidates = [group for group in groups if group[0] == gamma]
        count = max(1, int(round(0.1 * len(candidates))))
        chosen = generator.choice(len(candidates), size=count, replace=False)
        validation_groups.update(candidates[int(index)] for index in chosen)
    validation = [
        index for index, row in enumerate(metadata)
        if (float(row["gamma"]), int(row["seed"])) in validation_groups
    ]
    training = [
        index for index, row in enumerate(metadata)
        if (float(row["gamma"]), int(row["seed"])) not in validation_groups
    ]
    return torch.tensor(training), torch.tensor(validation)


def train(
    policy,
    contexts,
    plans,
    metadata,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    recovery_path: Path | None = None,
):
    training_ids, validation_ids = trajectory_split(metadata, seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0.03 * learning_rate,
    )
    history = []
    best_epoch = -1
    best_validation = float("inf")
    best_state = None
    for epoch in range(epochs):
        order = training_ids[
            torch.randperm(len(training_ids), generator=generator)
        ]
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            loss = policy.cfm_loss(
                contexts[indices].to(device),
                plans[indices].to(device),
                reduction="mean",
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        schedule.step()
        with torch.no_grad():
            validation_losses = []
            for start in range(0, len(validation_ids), 256):
                indices = validation_ids[start:start + 256]
                validation_losses.append(policy.cfm_loss(
                    contexts[indices].to(device),
                    plans[indices].to(device),
                    reduction="none",
                ).cpu())
            validation = float(torch.cat(validation_losses).mean())
        history.append({
            "epoch": epoch,
            "train": float(np.mean(losses)),
            "valid": validation,
        })
        if validation < best_validation:
            best_epoch = epoch
            best_validation = validation
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in policy.state_dict().items()
            }
            if recovery_path is not None:
                temporary = recovery_path.with_suffix(
                    recovery_path.suffix + ".tmp"
                )
                torch.save({
                    "epoch": best_epoch,
                    "validation_loss": best_validation,
                    "model": best_state,
                }, temporary)
                temporary.replace(recovery_path)
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(
                f"epoch {epoch:4d} train {history[-1]['train']:.4f} "
                f"valid {validation:.4f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    policy.load_state_dict(best_state, strict=True)
    return (
        history,
        training_ids,
        validation_ids,
        best_epoch,
        best_validation,
    )


def audit(policy, config, episodes: int, seed0: int) -> tuple[list[dict], list[dict]]:
    rows = []
    for gamma in config.data.gammas:
        for episode in range(episodes):
            result = raw_reference_rollout(
                policy,
                config,
                float(gamma),
                seed0 + 37 * episode,
            )
            rows.append({
                "gamma": float(gamma),
                "episode": int(episode),
                "seed": int(seed0 + 37 * episode),
                "status": result["status"],
                "mode": result["mode"],
                "window_validity": float(result["window_validity"]),
                "min_clearance_m": result["min_clearance_m"],
                "time_to_goal_s": result["time_to_goal_s"],
            })

    summaries = []
    for gamma in config.data.gammas:
        group = [row for row in rows if row["gamma"] == float(gamma)]
        successes = [row for row in group if row["status"] == "SUCCESS"]
        modes = {
            row["mode"] for row in successes if row["mode"] in LAB_ROUTE_MODES
        }
        counts = {
            mode: sum(row["mode"] == mode for row in successes)
            for mode in LAB_ROUTE_MODES
        }
        summaries.append({
            "gamma": float(gamma),
            "episodes": len(group),
            "SR": float(np.mean([row["status"] == "SUCCESS" for row in group])),
            "CR": float(np.mean([row["status"] == "COLLISION" for row in group])),
            "OOB": float(np.mean([row["status"] == "OOB" for row in group])),
            "timeout": float(np.mean([row["status"] == "TIMEOUT" for row in group])),
            "window_validity": float(np.mean([
                row["window_validity"] for row in group
            ])),
            "route_coverage": len(modes) / len(LAB_ROUTE_MODES),
            "route_counts": counts,
            "successful_min_clearance_m": (
                float(np.mean([
                    row["min_clearance_m"] for row in successes
                    if row["min_clearance_m"] is not None
                ]))
                if successes else None
            ),
            "successful_time_to_goal_s": (
                float(np.mean([
                    row["time_to_goal_s"] for row in successes
                    if row["time_to_goal_s"] is not None
                ]))
                if successes else None
            ),
        })
    return rows, summaries


def plot_results(history, summaries, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    axes[0].plot(
        [row["epoch"] for row in history],
        [row["train"] for row in history],
        label="train",
    )
    axes[0].plot(
        [row["epoch"] for row in history],
        [row["valid"] for row in history],
        label="trajectory-disjoint validation",
    )
    axes[0].set(xlabel="Epoch", ylabel="CFM loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    x = np.arange(len(summaries))
    width = 0.2
    metrics = (
        ("SR", "SR"),
        ("CR", "CR"),
        ("window_validity", "Window validity"),
        ("route_coverage", "Route coverage"),
    )
    for offset, (key, label) in enumerate(metrics):
        axes[1].bar(
            x + (offset - 1.5) * width,
            [row[key] for row in summaries],
            width,
            label=label,
        )
    axes[1].set_xticks(
        x,
        [rf"$\gamma={row['gamma']:g}$" for row in summaries],
    )
    axes[1].set_ylim(0.0, 1.03)
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=(
            ROOT
            / "results/lab_ball_pretrain/native_governed_w075_50pg_s0"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/lab_reference_flow",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--representation-dim", type=int, default=32)
    parser.add_argument("--grid-token-dim", type=int, default=32)
    parser.add_argument("--nfe", type=int, default=16)
    parser.add_argument(
        "--context-model",
        choices=("raw10", "visual_hp3d"),
        default="raw10",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        default="cpu",
    )
    parser.add_argument("--audit-episodes", type=int, default=50)
    parser.add_argument("--audit-seed", type=int, default=91000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    context_schema = (
        LAB_VISUAL_SCHEMA
        if args.context_model == "visual_hp3d"
        else LAB_RAW_CONTEXT_SCHEMA
    )
    contexts_np, plans_np, metadata, config = lab_reference_demo_windows(
        args.demo_dir,
        context_schema=context_schema,
    )
    expected_context_dim = (
        LAB_VISUAL_PACKED_DIM
        if args.context_model == "visual_hp3d"
        else LAB_REFERENCE_CONTEXT_DIM
    )
    if contexts_np.shape[1] != expected_context_dim:
        raise RuntimeError("lab reference context contract changed unexpectedly")
    contexts = torch.from_numpy(contexts_np)
    plans = torch.from_numpy(plans_np)
    torch.manual_seed(args.seed)
    if args.context_model == "visual_hp3d":
        policy = LabVisualFlowPolicy(
            plan_shape=(10, 3),
            hidden=args.hidden,
            representation_dim=args.representation_dim,
            grid_token_dim=args.grid_token_dim,
            control_limit=config.safemppi.demo_u_max,
            nfe=args.nfe,
            trunk_depth=2,
            time_features="raw1",
        )
    else:
        policy = ConditionalFlowMLP(
            context_dim=LAB_REFERENCE_CONTEXT_DIM,
            plan_shape=(10, 3),
            hidden=args.hidden,
            representation_dim=args.representation_dim,
            control_limit=config.safemppi.demo_u_max,
            nfe=args.nfe,
            trunk_depth=2,
            time_features="raw1",
        )
    device = torch.device(args.device)
    policy.to(device)
    print(
        f"[dataset] {len(contexts)} windows from "
        f"{len({(row['gamma'], row['seed']) for row in metadata})} trajectories",
        flush=True,
    )
    (
        history,
        training_ids,
        validation_ids,
        best_epoch,
        best_validation,
    ) = train(
        policy,
        contexts,
        plans,
        metadata,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
        recovery_path=args.output / ".best_training_state.pt",
    )

    generator = torch.Generator().manual_seed(args.seed + 1)
    calibration_ids = torch.randperm(
        len(contexts), generator=generator,
    )[:50]
    calibration = policy.embed(
        contexts[calibration_ids].to(device),
        plans[calibration_ids].to(device),
    ).cpu()
    lengthscale = mean_pairwise_lengthscale(calibration)
    policy.cpu()
    rows, summaries = audit(
        policy,
        config,
        args.audit_episodes,
        args.audit_seed,
    )

    if args.context_model == "visual_hp3d":
        arch = {
            "kind": LAB_VISUAL_SCHEMA,
            "plan_shape": [10, 3],
            "hidden": args.hidden,
            "representation_dim": args.representation_dim,
            "grid_token_dim": args.grid_token_dim,
            "grid_shape": list(LAB_VISUAL_GRID_SHAPE),
            "grid_channels": list(LAB_VISUAL_CHANNELS),
            "grid_frame": LAB_VISUAL_FRAME,
            "control_limit": config.safemppi.demo_u_max,
            "nfe": args.nfe,
            "trunk_depth": 2,
            "time_features": "raw1",
        }
    else:
        arch = {
            "kind": "conditional_flow_mlp",
            "context_dim": LAB_REFERENCE_CONTEXT_DIM,
            "plan_shape": [10, 3],
            "hidden": args.hidden,
            "representation_dim": args.representation_dim,
            "control_limit": config.safemppi.demo_u_max,
            "nfe": args.nfe,
            "trunk_depth": 2,
            "time_features": "raw1",
        }
    checkpoint = {
        "model": policy.state_dict(),
        "arch": arch,
        "contract": {
            "policy_output": "pre_smoothing_raw_acceleration_command",
            "stateful_governor_in_policy": False,
            "deployment_smoothing_and_tracking": "external",
        },
    }
    torch.save(checkpoint, args.output / "pretrained.pt")
    torch.save(calibration, args.output / "calibration_features.pt")
    manifest = {
        "kind": "lab raw-command reference-flow pretraining",
        "source_demo_dir": str(args.demo_dir.resolve()),
        "context_model": args.context_model,
        "context_schema": context_schema,
        "external_context_dim": expected_context_dim,
        "encoded_context_dim": (
            7 + args.grid_token_dim
            if args.context_model == "visual_hp3d"
            else LAB_REFERENCE_CONTEXT_DIM
        ),
        "policy_output": "pre_smoothing_raw_acceleration_command",
        "stateful_governor_in_policy": False,
        "deployment_smoothing_and_tracking": "external",
        "windows": len(contexts),
        "training_windows": len(training_ids),
        "validation_windows": len(validation_ids),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "final_train_loss": history[-1]["train"],
        "final_valid_loss": history[-1]["valid"],
        "selected_epoch": best_epoch,
        "selected_valid_loss": best_validation,
        "checkpoint_selection": "minimum_trajectory_disjoint_validation_loss",
        "rbf_lengthscale": lengthscale,
        "raw_temperature": 1.0,
        "raw_audit_episodes_per_gamma": args.audit_episodes,
        "raw_audit_seed": args.audit_seed,
        "raw_audit_summary": summaries,
        "raw_audit": rows,
    }
    (args.output / "pretrain_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    plot_results(
        history,
        summaries,
        args.output / "pretrain_qualification.png",
    )
    (args.output / ".best_training_state.pt").unlink(missing_ok=True)
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"[output] {args.output}", flush=True)


if __name__ == "__main__":
    main()
