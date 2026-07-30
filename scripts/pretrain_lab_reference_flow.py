"""Pretrain and qualify the Minhyuk-frame reference flow policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.config import load_config
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
    LAB_VISUAL_HISTORY_LENGTH,
    LAB_VISUAL_HISTORY_PACKED_DIM,
    LAB_VISUAL_HISTORY_SCHEMA,
    LAB_VISUAL_PACKED_DIM,
    LAB_VISUAL_SCHEMA,
    LabVisualHistoryFlowPolicy,
    LabVisualFlowPolicy,
)


ROOT = Path(__file__).resolve().parents[1]


def source_archive_digest(run_dir: str | Path) -> dict:
    """Hash the manifest, resolved config, and every manifest-declared run."""
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    relative_names = {"manifest.json", "resolved_config.json"}
    relative_names.update(
        str(row["file"])
        for row in manifest.get("runs", ())
        if row.get("file") is not None
    )

    digest = hashlib.sha256()
    digest.update(b"safe_mppi_declared_demo_archive_v1\0")
    for name in sorted(relative_names):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"archive file must remain under demo dir: {name}")
        path = (run_dir / relative).resolve()
        try:
            path.relative_to(run_dir)
        except ValueError as error:
            raise ValueError(
                f"archive file must remain under demo dir: {name}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"archive provenance file missing: {path}")
        encoded_name = relative.as_posix().encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "algorithm": "sha256",
        "schema": "safe_mppi_declared_demo_archive_v1",
        "sha256": digest.hexdigest(),
        "file_count": len(relative_names),
    }


def split_provenance(
    metadata: list[dict],
    training_ids: torch.Tensor,
    validation_ids: torch.Tensor,
    seed: int,
) -> dict:
    """Return reproducible split metadata for randomized and legacy archives."""
    all_scene_hashes = sorted({
        str(row["scene_hash"])
        for row in metadata
        if row.get("scene_hash") is not None
    })

    def selected_scene_hashes(indices: torch.Tensor) -> list[str]:
        return sorted({
            str(metadata[int(index)]["scene_hash"])
            for index in indices.tolist()
            if metadata[int(index)].get("scene_hash") is not None
        })

    return {
        "split_seed": int(seed),
        "split_unit": (
            "scene_sha256_across_all_gamma"
            if all_scene_hashes else "gamma_and_trajectory_seed"
        ),
        "randomized_scene_count": len(all_scene_hashes),
        "training_scene_hashes": selected_scene_hashes(training_ids),
        "validation_scene_hashes": selected_scene_hashes(validation_ids),
    }


@torch.no_grad()
def pretrained_sample_calibration(
    policy,
    contexts: torch.Tensor,
    metadata: list[dict],
    training_ids: torch.Tensor,
    *,
    seed: int,
    count: int = 50,
) -> tuple[torch.Tensor, list[int]]:
    """Embed 50 pretrained-policy samples on gamma-balanced train contexts."""
    if count < 2:
        raise ValueError("RBF calibration requires at least two samples")
    training = np.asarray(training_ids.tolist(), dtype=np.int64)
    rng = np.random.default_rng(int(seed) + 1)
    gammas = sorted({
        float(metadata[int(index)]["gamma"]) for index in training
    })
    selected: list[int] = []
    base_count, remainder = divmod(int(count), len(gammas))
    for gamma_index, gamma in enumerate(gammas):
        target = base_count + int(gamma_index < remainder)
        available = training[[
            np.isclose(
                float(metadata[int(index)]["gamma"]),
                gamma,
                rtol=0.0,
                atol=1.0e-7,
            )
            for index in training
        ]]
        if len(available) < target:
            raise ValueError(
                f"RBF calibration needs {target} training contexts "
                f"for gamma={gamma:g}, found {len(available)}"
            )
        selected.extend(
            rng.choice(available, size=target, replace=False).tolist()
        )
    selected_contexts = contexts[torch.tensor(selected, dtype=torch.long)]
    generator = torch.Generator().manual_seed(int(seed) + 2)
    plans, bases = [], []
    for context in selected_contexts:
        plan, base = policy.sample_with_base(
            context, 1, generator, base_std=1.0,
        )
        plans.append(plan[0])
        bases.append(base[0])
    features = policy.embed(
        selected_contexts,
        torch.stack(plans),
        base=torch.stack(bases),
    )
    if (
        features.shape[0] != count
        or not bool(torch.isfinite(features).all())
    ):
        raise RuntimeError(
            "pretrained-policy RBF calibration produced invalid features"
        )
    return features.cpu(), selected


def trajectory_split(metadata: list[dict], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Disjoint train/validation split over whole demonstration trajectories."""
    randomized_scenes = {
        str(row["scene_hash"])
        for row in metadata
        if row.get("scene_hash") is not None
    }
    generator = np.random.default_rng(seed)
    if randomized_scenes:
        if any(row.get("scene_hash") is None for row in metadata):
            raise ValueError(
                "randomized archive mixes rows with and without scene hashes"
            )
        scenes = sorted(randomized_scenes)
        if len(scenes) < 2:
            raise ValueError(
                "randomized pretraining requires at least two distinct scenes"
            )
        count = max(1, int(round(0.1 * len(scenes))))
        chosen = generator.choice(len(scenes), size=count, replace=False)
        validation_scenes = {scenes[int(index)] for index in chosen}
        validation = [
            index for index, row in enumerate(metadata)
            if str(row.get("scene_hash")) in validation_scenes
        ]
        training = [
            index for index, row in enumerate(metadata)
            if str(row.get("scene_hash")) not in validation_scenes
        ]
        if not training or not validation:
            raise ValueError(
                "scene-disjoint split produced an empty train or validation set"
            )
        return torch.tensor(training), torch.tensor(validation)

    groups = sorted({
        (
            float(row["gamma"]),
            str(
                row.get("scene_id")
                if row.get("scene_id") is not None
                else row["seed"]
            ),
        )
        for row in metadata
    })
    validation_groups = set()
    for gamma in sorted({group[0] for group in groups}):
        candidates = [group for group in groups if group[0] == gamma]
        count = max(1, int(round(0.1 * len(candidates))))
        chosen = generator.choice(len(candidates), size=count, replace=False)
        validation_groups.update(candidates[int(index)] for index in chosen)
    validation = [
        index for index, row in enumerate(metadata)
        if (
            float(row["gamma"]),
            str(
                row.get("scene_id")
                if row.get("scene_id") is not None
                else row["seed"]
            ),
        ) in validation_groups
    ]
    training = [
        index for index, row in enumerate(metadata)
        if (
            float(row["gamma"]),
            str(
                row.get("scene_id")
                if row.get("scene_id") is not None
                else row["seed"]
            ),
        ) not in validation_groups
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


def audit(
    policy,
    config,
    episodes: int,
    seed0: int,
    *,
    device: str | torch.device = "cpu",
) -> tuple[list[dict], list[dict]]:
    scene_randomization = config.raw.get("scene_randomization", {})
    clutter_scenes = None
    if scene_randomization.get("enabled"):
        if scene_randomization.get("distribution") == (
            "path_focused_truncated_normal_v1"
        ):
            from safe_mppi.path_focused_clutter import (
                path_focused_scene_bank,
            )
            clutter_scenes = path_focused_scene_bank(
                config, episodes, seed=seed0,
            )
        else:
            if scene_randomization.get("obstacle_family") != "vertical_cylinders":
                raise ValueError(
                    "legacy pretraining qualification supports cylinders only"
                )
            from safe_mppi.lab_clutter import cylinder_scene_bank
            clutter_scenes = cylinder_scene_bank(
                episodes,
                seed=seed0,
                bounds=config.taskspace.bounds,
                start=np.asarray(config.taskspace.start),
                goal=np.asarray(config.taskspace.goal),
                cylinder_count=int(scene_randomization["count"]),
                cylinder_radius_m=float(scene_randomization["radius_m"]),
                min_surface_gap_m=float(
                    scene_randomization["minimum_obstacle_surface_gap_m"]
                ),
                start_surface_gap_m=float(
                    scene_randomization["minimum_start_surface_clearance_m"]
                ),
                goal_surface_gap_m=float(
                    scene_randomization["minimum_goal_surface_clearance_m"]
                ),
                boundary_surface_gap_m=float(
                    scene_randomization[
                        "minimum_taskspace_wall_surface_clearance_m"
                    ]
                ),
            )
    randomized_clutter = clutter_scenes is not None
    if clutter_scenes is not None:
        from safe_mppi.lab_clutter import config_for_scene
    rollout_kwargs = (
        {}
        if torch.device(device).type == "cpu"
        else {"device": device}
    )
    rows = []
    for gamma in config.data.gammas:
        for episode in range(episodes):
            rollout_config = (
                config_for_scene(config, clutter_scenes[episode])
                if clutter_scenes is not None else config
            )
            result = raw_reference_rollout(
                policy,
                rollout_config,
                float(gamma),
                seed0 + 37 * episode,
                **rollout_kwargs,
            )
            rows.append({
                "gamma": float(gamma),
                "episode": int(episode),
                "seed": int(seed0 + 37 * episode),
                "scene_id": (
                    clutter_scenes[episode].scene_id
                    if clutter_scenes is not None else None
                ),
                "scene_hash": (
                    clutter_scenes[episode].scene_hash
                    if clutter_scenes is not None else None
                ),
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
        counts = (
            None
            if randomized_clutter else {
                mode: sum(row["mode"] == mode for row in successes)
                for mode in LAB_ROUTE_MODES
            }
        )
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
            "route_coverage": (
                None
                if randomized_clutter
                else len(modes) / len(LAB_ROUTE_MODES)
            ),
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


def plot_results(
    history,
    summaries,
    output: Path,
    ood_summaries=None,
) -> None:
    columns = 3 if ood_summaries is not None else 2
    fig, axes = plt.subplots(1, columns, figsize=(5.6 * columns, 4.1))
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

    def metric_bars(axis, values, title):
        x = np.arange(len(values))
        width = 0.2
        metrics = [
            ("SR", "SR"),
            ("CR", "CR"),
            ("window_validity", "Window validity"),
            ("OOB", "OOB"),
        ]
        for offset, (key, label) in enumerate(metrics):
            axis.bar(
                x + (offset - 1.5) * width,
                [row[key] for row in values],
                width,
                label=label,
            )
        axis.set_xticks(
            x,
            [rf"$\gamma={row['gamma']:g}$" for row in values],
        )
        axis.set_ylim(0.0, 1.03)
        axis.set_title(title)
        axis.grid(alpha=0.25, axis="y")
        axis.legend(fontsize=8)

    metric_bars(axes[1], summaries, "Cylinder ID")
    if ood_summaries is not None:
        metric_bars(axes[2], ood_summaries, "Sphere OOD")
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
        choices=("raw10", "visual_hp3d", "visual_hp3d_gru"),
        default="raw10",
    )
    parser.add_argument("--history-token-dim", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--audit-episodes", type=int, default=50)
    parser.add_argument("--audit-seed", type=int, default=91000)
    parser.add_argument(
        "--ood-config",
        type=Path,
        default=None,
        help="optional path-focused sphere config for an OOD raw audit",
    )
    parser.add_argument("--ood-audit-episodes", type=int, default=None)
    parser.add_argument("--ood-audit-seed", type=int, default=191000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    context_schema = {
        "raw10": LAB_RAW_CONTEXT_SCHEMA,
        "visual_hp3d": LAB_VISUAL_SCHEMA,
        "visual_hp3d_gru": LAB_VISUAL_HISTORY_SCHEMA,
    }[args.context_model]
    contexts_np, plans_np, metadata, config = lab_reference_demo_windows(
        args.demo_dir,
        context_schema=context_schema,
    )
    archive_digest = source_archive_digest(args.demo_dir)
    expected_context_dim = {
        "raw10": LAB_REFERENCE_CONTEXT_DIM,
        "visual_hp3d": LAB_VISUAL_PACKED_DIM,
        "visual_hp3d_gru": LAB_VISUAL_HISTORY_PACKED_DIM,
    }[args.context_model]
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
    elif args.context_model == "visual_hp3d_gru":
        policy = LabVisualHistoryFlowPolicy(
            plan_shape=(10, 3),
            hidden=args.hidden,
            representation_dim=args.representation_dim,
            grid_token_dim=args.grid_token_dim,
            history_token_dim=args.history_token_dim,
            history_length=LAB_VISUAL_HISTORY_LENGTH,
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

    policy.cpu()
    calibration, calibration_ids = pretrained_sample_calibration(
        policy,
        contexts,
        metadata,
        training_ids,
        seed=args.seed,
    )
    lengthscale = mean_pairwise_lengthscale(calibration)
    policy.to(device)
    rows, summaries = audit(
        policy,
        config,
        args.audit_episodes,
        args.audit_seed,
        device=device,
    )
    demo_manifest = json.loads(
        (args.demo_dir / "manifest.json").read_text()
    )
    training_scene_hashes = {
        str(row["scene_hash"])
        for row in demo_manifest.get("scene_bank", {}).get("scenes", [])
        if row.get("scene_hash") is not None
    }
    id_audit_scene_hashes = {
        str(row["scene_hash"])
        for row in rows if row.get("scene_hash") is not None
    }
    id_overlap = training_scene_hashes & id_audit_scene_hashes
    if id_overlap:
        raise RuntimeError(
            "raw ID audit scene bank overlaps the demonstration geometry bank"
        )
    ood_rows = None
    ood_summaries = None
    if args.ood_config is not None:
        ood_config = load_config(args.ood_config)
        ood_rows, ood_summaries = audit(
            policy,
            ood_config,
            (
                args.audit_episodes
                if args.ood_audit_episodes is None
                else args.ood_audit_episodes
            ),
            args.ood_audit_seed,
            device=device,
        )
    policy.cpu()

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
    elif args.context_model == "visual_hp3d_gru":
        arch = {
            "kind": LAB_VISUAL_HISTORY_SCHEMA,
            "plan_shape": [10, 3],
            "hidden": args.hidden,
            "representation_dim": args.representation_dim,
            "grid_token_dim": args.grid_token_dim,
            "history_token_dim": args.history_token_dim,
            "history_length": LAB_VISUAL_HISTORY_LENGTH,
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
            "past_raw_action_history_in_policy": (
                args.context_model == "visual_hp3d_gru"
            ),
            "deployment_smoothing_and_tracking": "external",
        },
    }
    torch.save(checkpoint, args.output / "pretrained.pt")
    torch.save(calibration, args.output / "calibration_features.pt")
    split_details = split_provenance(
        metadata,
        training_ids,
        validation_ids,
        args.seed,
    )
    manifest = {
        "kind": "lab raw-command reference-flow pretraining",
        "source_demo_dir": str(args.demo_dir.resolve()),
        "source_archive_digest": archive_digest,
        "context_model": args.context_model,
        "context_schema": context_schema,
        "external_context_dim": expected_context_dim,
        "encoded_context_dim": (
            7 + args.grid_token_dim + args.history_token_dim
            if args.context_model == "visual_hp3d_gru"
            else (
                7 + args.grid_token_dim
                if args.context_model == "visual_hp3d"
                else LAB_REFERENCE_CONTEXT_DIM
            )
        ),
        "policy_output": "pre_smoothing_raw_acceleration_command",
        "stateful_governor_in_policy": False,
        "past_raw_action_history_in_policy": (
            args.context_model == "visual_hp3d_gru"
        ),
        "history_length": (
            LAB_VISUAL_HISTORY_LENGTH
            if args.context_model == "visual_hp3d_gru" else 0
        ),
        "history_token_dim": (
            args.history_token_dim
            if args.context_model == "visual_hp3d_gru" else 0
        ),
        "history_encoder_must_freeze_during_expansion": (
            args.context_model == "visual_hp3d_gru"
        ),
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
        **split_details,
        "rbf_lengthscale": lengthscale,
        "rbf_calibration": {
            "source": "pretrained_policy_samples_on_training_contexts",
            "count": len(calibration_ids),
            "gamma_balanced": True,
            "flow_time": 0.9,
            "paired_base_noise": True,
            "context_indices": calibration_ids,
            "scene_hashes": sorted({
                str(metadata[index]["scene_hash"])
                for index in calibration_ids
                if metadata[index].get("scene_hash") is not None
            }),
        },
        "raw_temperature": 1.0,
        "raw_audit_episodes_per_gamma": args.audit_episodes,
        "raw_audit_seed": args.audit_seed,
        "raw_audit_scene_overlap_count": len(id_overlap),
        "raw_audit_scene_bank_disjoint_from_demo": True,
        "raw_audit_summary": summaries,
        "raw_audit": rows,
        "ood_raw_audit_config": (
            str(args.ood_config.resolve())
            if args.ood_config is not None else None
        ),
        "ood_raw_audit_episodes_per_gamma": (
            args.audit_episodes
            if args.ood_config is not None
            and args.ood_audit_episodes is None
            else args.ood_audit_episodes
        ),
        "ood_raw_audit_seed": (
            args.ood_audit_seed if args.ood_config is not None else None
        ),
        "ood_raw_audit_summary": ood_summaries,
        "ood_raw_audit": ood_rows,
    }
    (args.output / "pretrain_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    plot_results(
        history,
        summaries,
        args.output / "pretrain_qualification.png",
        ood_summaries,
    )
    (args.output / ".best_training_state.pt").unlink(missing_ok=True)
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"[output] {args.output}", flush=True)


if __name__ == "__main__":
    main()
