#!/usr/bin/env python3
"""Build the winner committed/PRE/R3 bowling interactive 3-D comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment
from safe_mppi.lab_clutter_expansion import sphere_scene_spec_from_config


def _encode_path(path: np.ndarray, limit: int = 44) -> list[Any]:
    values = np.asarray(path, np.float64).reshape(-1, 3)
    if len(values) > limit:
        keep = np.unique(np.linspace(0, len(values) - 1, limit).round().astype(int))
        values = values[keep]
    quantized = np.rint(values * 1000.0).astype(np.int32)
    deltas = np.diff(quantized, axis=0)
    return [quantized[0].tolist(), deltas.reshape(-1).tolist()]


def _committed(expansion: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((expansion / "manifest.json").read_text())
    config = load_config(expansion / "task_config_resolved.json")
    env = TaskEnvironment(config)
    scene_spec = sphere_scene_spec_from_config(config)
    rows: list[dict[str, Any]] = []
    round_summary: list[dict[str, Any]] = []
    cumulative = 0
    for detail in manifest["rounds"]:
        round_i = int(detail["round"])
        events = torch.load(
            expansion / f"events_round_{round_i:03d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        by_key: dict[tuple[float, int], list[dict[str, Any]]] = {}
        for event in events:
            key = (float(event["gamma"]), int(event["episode"]))
            by_key.setdefault(key, []).append(event)
        committed_count = 0
        by_gamma: dict[str, int] = {}
        for gamma_raw, committed in detail["successful_executed_commit_by_gamma"].items():
            gamma = float(gamma_raw)
            episode_ids = [int(value) for value in committed["committed_episode_ids"]]
            by_gamma[f"{gamma:g}"] = len(episode_ids)
            for episode in episode_ids:
                trajectory = sorted(by_key[(gamma, episode)], key=lambda row: int(row["step"]))
                if not trajectory or trajectory[-1].get("status") != "SUCCESS":
                    raise RuntimeError(
                        f"committed r{round_i} gamma={gamma:g} episode={episode} is not terminal-success"
                    )
                path = np.vstack([
                    np.asarray(trajectory[0]["robot"], np.float64)[:3],
                    *[
                        np.asarray(event["robot_after"], np.float64)[:3]
                        for event in trajectory
                    ],
                ])
                raw_context = trajectory[0]["context"]
                if isinstance(raw_context, torch.Tensor):
                    raw_context = raw_context.detach().cpu().numpy()
                context = np.asarray(raw_context, np.float32).reshape(-1)
                spheres = scene_spec.unpack(env, context[-scene_spec.packed_dim:])
                first = trajectory[0]
                rows.append({
                    "id": f"r{round_i}:g{gamma:g}:e{episode}",
                    "kind": "committed",
                    "round": round_i,
                    "gamma": gamma,
                    "episode": episode,
                    "status": "SUCCESS",
                    "route": None,
                    "path": _encode_path(path),
                    "spheres": np.round(spheres, 4).tolist(),
                    "steps": len(trajectory),
                    "sceneHash": str(first.get("scene_hash", ""))[:12],
                    "member": first.get("paired_scene_member_name"),
                    "pairSlot": first.get("paired_scene_pair_slot"),
                })
                committed_count += 1
        cumulative += committed_count
        round_summary.append({
            "round": round_i,
            "count": committed_count,
            "cumulative": cumulative,
            "byGamma": by_gamma,
            "optimizerStepsThisRound": int(detail["steps"]),
            "optimizerStep": int(detail["optimizer_step"]),
            "loss": round(float(detail["positive_loss"]), 6),
        })
    if cumulative != 146:
        raise RuntimeError(f"winner committed count changed: expected 146, found {cumulative}")
    return rows, round_summary


def _bowling(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    stable_codes = {"LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR"}
    for round_raw, values in payload["rows"].items():
        round_i = int(round_raw)
        checkpoint = "PRE" if round_i == 0 else f"R{round_i}"
        for value in values:
            route = value.get("bowling_route")
            stable_route = route.get("stable_code") if isinstance(route, dict) else None
            rows.append({
                "id": f"{checkpoint}:g{float(value['gamma']):g}:e{int(value['episode'])}",
                "kind": "bowling",
                "checkpoint": checkpoint,
                "round": round_i,
                "gamma": float(value["gamma"]),
                "episode": int(value["episode"]),
                "status": str(value["status"]),
                "route": stable_route if stable_route in stable_codes else None,
                "rawRoute": route.get("code") if isinstance(route, dict) else None,
                "vertical": (
                    route.get("decision_vertical_sign") if isinstance(route, dict) else None
                ),
                "path": _encode_path(np.asarray(value["path_xyz_m"], np.float64)),
                "spheres": None,
                "steps": len(value["path_xyz_m"]),
                "clearance": value.get("min_clearance_m"),
                "ttg": value.get("time_to_goal_s"),
                "validity": value.get("window_validity"),
            })
    return rows, {
        "scene": payload["scene"],
        "summary": payload["summary"],
        "deployment": payload["deployment_contract"],
        "binding": payload["artifact_binding"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--bowling-eval", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-output", type=Path)
    args = parser.parse_args()

    committed, rounds = _committed(args.expansion)
    bowling, bowling_meta = _bowling(args.bowling_eval)
    config = load_config(args.expansion / "task_config_resolved.json")
    env = TaskEnvironment(config)
    payload = {
        "start": np.round(env.start[:3], 5).tolist(),
        "goal": np.round(env.goal[:3], 5).tolist(),
        "bounds": np.round(env.bounds, 5).tolist(),
        "rounds": rounds,
        "committed": committed,
        "bowling": bowling,
        "bowlingMeta": bowling_meta,
    }
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    if args.data_output is not None:
        args.data_output.parent.mkdir(parents=True, exist_ok=True)
        args.data_output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    template = args.template.read_text()
    marker = "__WINNER_BOWLING_DATA__"
    if template.count(marker) != 1:
        raise RuntimeError("visual template must contain exactly one data marker")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace(marker, encoded))
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "committed": len(committed),
        "bowling": len(bowling),
    }, indent=2))


if __name__ == "__main__":
    main()
