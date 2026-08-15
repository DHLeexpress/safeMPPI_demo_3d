#!/usr/bin/env python3
"""Append five as-built gamma=0.3 views to the existing paper-ready site."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


BUNDLE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BUNDLE / "source"), str(BUNDLE / "runtime_snapshot")]

from build_paper_ready_bowling_handoff import (  # noqa: E402
    ROUTES,
    _annotate,
    _outer_document,
    _viz_row,
)
from real_bowling_scene import load_as_built_geometry  # noqa: E402


GAMMA = 0.3
Z_MARGIN_M = 0.3
VIEW_NAMES = (
    "real-paper-ready-pre2",
    "real-paper-ready-less-expanded",
    "real-paper-ready-expanded",
    "real-paper-ready-cfmmppi",
    "real-paper-ready-safemppi",
)

SELECTION_NOTES = {
    "real-paper-ready-pre2": "Narrative roster: LLL×6 + RRR×2.",
    "real-paper-ready-less-expanded": "Four-mode R1 roster: LLL / LLR / RLL / RRR.",
    "real-paper-ready-expanded": (
        "S4 roster with monotone LLR seed 814323086 replacing 814321091."
    ),
    "real-paper-ready-cfmmppi": (
        "Safety/balanced successes plus two string-safe reward collisions."
    ),
    "real-paper-ready-safemppi": (
        "Lateral-route roster: LLL / LLR / RRL / RRR, two each."
    ),
}

EXPANDED_SEEDS = (
    814320088,
    814321275,
    814321225,
    814323086,  # monotone/smoother LLR replacement for 814321091
    814321181,
    814324192,
    814321156,
    814321368,
)


def _load_policy(path: Path) -> list[dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a rollout list")
    return payload


def _dynamic_quality(rows: list[dict], goal, z_low: float, z_high: float) -> list[dict]:
    rows = [
        {
            **row,
            "episode": int(row.get("episode", row.get("trial", 0))),
        }
        for row in rows
    ]
    annotated = _annotate(rows, goal)
    for row in annotated:
        states = np.asarray(row["states"], float)
        occupancy = float(np.mean(
            (states[:, 2] >= z_low) & (states[:, 2] <= z_high)
        ))
        quality = row["paper_quality"]
        quality["z_band_low_m"] = float(z_low)
        quality["z_band_high_m"] = float(z_high)
        quality["z_band_occupancy"] = occupancy
        quality["hard_z_pass"] = occupancy >= 0.9
        hard = row.get("hard_constraints") or {}
        quality["effective_sphere_min_clearance_m"] = hard.get(
            "effective_sphere_min_clearance_m"
        )
        quality["string_min_clearance_m"] = hard.get("string_min_clearance_m")
        quality["hard_real_geometry_pass"] = bool(hard.get("hard_valid", False))
        route = row.get("bowling_route") or {}
        row["stable_route"] = (
            route.get("stable_code") if route.get("stable_code") in ROUTES else None
        )
    return annotated


def _eligible(rows: list[dict], *, goal_progress: bool) -> list[dict]:
    return [
        row for row in rows
        if np.isclose(float(row["gamma"]), GAMMA)
        and row["status"] == "SUCCESS"
        and row["stable_route"] in ROUTES
        and row["paper_quality"]["hard_z_pass"]
        and row["paper_quality"]["hard_real_geometry_pass"]
        and (
            not goal_progress
            or row["paper_quality"]["hard_goal_progress_pass"]
        )
    ]


def _rank(row: dict) -> tuple:
    quality = row["paper_quality"]
    return (
        quality["quality_score"],
        quality["decision_mean_abs_z_m"],
        quality["mean_curvature_rad_per_m"],
        int(row.get("episode", row.get("trial", 0))),
    )


def _balanced(
    rows: list[dict],
    *,
    count: int = 8,
    maximum_modes: int | None = None,
) -> list[dict]:
    by_route = {
        route: sorted(
            [row for row in rows if row["stable_route"] == route],
            key=_rank,
        )
        for route in ROUTES
    }
    available = [route for route in ROUTES if by_route[route]]
    available.sort(key=lambda route: (_rank(by_route[route][0]), route))
    if maximum_modes is not None:
        # Preserve diversity while keeping the requested PRE2/R1 mode ceiling.
        available = available[:maximum_modes]
    selected = []
    depth = 0
    while len(selected) < count:
        added = False
        for route in available:
            if depth < len(by_route[route]):
                selected.append(by_route[route][depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != count:
        raise ValueError(
            f"only {len(selected)}/{count} selectable trajectories; "
            f"available modes={available}"
        )
    return selected


def _route_quota(rows: list[dict], quotas: dict[str, int]) -> list[dict]:
    selected = []
    for route, count in quotas.items():
        candidates = sorted(
            [row for row in rows if row["stable_route"] == route],
            key=_rank,
        )
        if len(candidates) < count:
            raise ValueError(
                f"route {route} has only {len(candidates)}/{count} rows"
            )
        selected.extend(candidates[:count])
    return selected


def _select_seeds(rows: list[dict], seeds: tuple[int, ...]) -> list[dict]:
    by_seed = {int(row["rollout_seed"]): row for row in rows}
    missing = [seed for seed in seeds if seed not in by_seed]
    if missing:
        raise ValueError(f"selected rollout seeds are not eligible: {missing}")
    return [by_seed[seed] for seed in seeds]


def _balanced_cfm(rows: list[dict]) -> list[dict]:
    """Keep the three intended CFM regimes in an eight-trajectory panel.

    Safety/balanced rows must be effective-shell-valid successes.  Reward rows
    intentionally retain the baseline's least-severe obstacle collisions, but
    every displayed row must obey the vertical-string hard gate.
    """
    selected = []
    quotas = {"safety_dominant": 3, "balanced": 3, "reward_dominant": 2}
    for regime, quota in quotas.items():
        candidates = [
            row for row in rows
            if row.get("regime") == regime
            and np.isclose(float(row["gamma"]), GAMMA)
            and row["paper_quality"]["hard_z_pass"]
            and (row.get("hard_constraints") or {}).get("string_valid", False)
        ]
        if regime != "reward_dominant":
            candidates = [
                row for row in candidates
                if row["status"] == "SUCCESS"
                and row["paper_quality"]["hard_real_geometry_pass"]
                and row["stable_route"] in ROUTES
            ]
            candidates.sort(key=_rank)
        else:
            candidates = [row for row in candidates if row["status"] == "COLLISION"]
            candidates.sort(key=lambda row: (
                -float((row.get("hard_constraints") or {}).get(
                    "effective_sphere_min_clearance_m", -float("inf"),
                )),
                _rank(row),
            ))
        if len(candidates) < quota:
            raise ValueError(
                f"CFM--MPPI {regime} has only {len(candidates)}/{quota} rows"
            )
        selected.extend(candidates[:quota])
    return selected


def _extract_data(inner: str) -> tuple[dict, int, int]:
    start = inner.index("const DATA=") + len("const DATA=")
    end = inner.index(",group=document.querySelector", start)
    return json.loads(inner[start:end]), start, end


def _patch_javascript(inner: str) -> str:
    inner = inner.replace(
        "function plane(z,opacity=.035){const b=DATA.scene.bounds;",
        """function activeScene(){return group.value.startsWith('real-paper-ready-')?DATA.realScene:DATA.scene}
function sphereLayer(s,color,opacity,name){const[cx,cy,cz,r]=s,x=[],y=[],z=[];for(let a=0;a<=10;a++){const th=Math.PI*a/10,X=[],Y=[],Z=[];for(let b=0;b<=18;b++){const ph=2*Math.PI*b/18;X.push(cx+r*Math.sin(th)*Math.cos(ph));Y.push(cy+r*Math.sin(th)*Math.sin(ph));Z.push(cz+r*Math.cos(th))}x.push(X);y.push(Y);z.push(Z)}return{type:'surface',x,y,z,showscale:false,hoverinfo:'skip',opacity,colorscale:[[0,color],[1,color]],name}}
function stringLayer(c,i){const[cx,cy,z0,r]=c,b=activeScene().bounds,x=[],y=[],z=[];for(let a=0;a<=1;a++){const X=[],Y=[],Z=[];for(let k=0;k<=24;k++){const p=2*Math.PI*k/24;X.push(cx+r*Math.cos(p));Y.push(cy+r*Math.sin(p));Z.push(a?b[2][1]:z0)}x.push(X);y.push(Y);z.push(Z)}return{type:'surface',x,y,z,showscale:false,hoverinfo:'skip',opacity:.22,colorscale:[[0,'#27272a'],[1,'#27272a']],name:`string no-go ${i+1}`}}
function obstacleTraces(){const s=activeScene();if(!group.value.startsWith('real-paper-ready-'))return s.spheres.map(sphere);return[...s.designPhysicalSpheres.map((v,i)=>sphereLayer(v,'#a1a1aa',.06,`old physical ${i+1}`)),...s.designEffectiveSpheres.map((v,i)=>sphereLayer(v,'#71717a',.09,`old effective ${i+1}`)),...s.effectiveSpheres.map((v,i)=>sphereLayer(v,'#f59e0b',.14,`real physical+0.16 ${i+1}`)),...s.physicalSpheres.map((v,i)=>sphereLayer(v,'#cc6e5c',.48,`real physical ${i+1}`)),...s.strings.map(stringLayer)]}
function plane(z,opacity=.035){const b=activeScene().bounds;""",
    )
    inner = inner.replace(
        "(group.value.startsWith('paper-ready-')?gs:",
        "((group.value.startsWith('paper-ready-')||group.value.startsWith('real-paper-ready-'))?gs:",
    )
    old = "const traces=[plane(DATA.audit.hard_z_contract.band_m[0]),plane(.9,.075),plane(DATA.audit.hard_z_contract.band_m[1]),...DATA.scene.spheres.map(sphere),"
    new = "const sc=activeScene(),band=group.value.startsWith('real-paper-ready-')?DATA.audit.real.hard_z_band_m:DATA.audit.hard_z_contract.band_m;const traces=[plane(band[0]),plane(.9,.075),plane(band[1]),...obstacleTraces(),"
    inner = inner.replace(old, new)
    inner = inner.replace(
        "x:[DATA.scene.start[0],DATA.scene.goal[0]],y:[DATA.scene.start[1],DATA.scene.goal[1]],z:[DATA.scene.start[2],DATA.scene.goal[2]]",
        "x:[sc.start[0],sc.goal[0]],y:[sc.start[1],sc.goal[1]],z:[sc.start[2],sc.goal[2]]",
    )
    inner = inner.replace(
        "range:DATA.scene.bounds[0]", "range:sc.bounds[0]",
    ).replace("range:DATA.scene.bounds[1]", "range:sc.bounds[1]").replace(
        "range:DATA.scene.bounds[2]", "range:sc.bounds[2]",
    )
    inner = inner.replace(
        "if(group.value==='paper-ready-less-expanded')",
        """if(group.value.startsWith('real-paper-ready-')){const a=DATA.audit.real,g=a.groups[group.value],cfm=group.value==='real-paper-ready-cfmmppi';metrics.innerHTML=[['Shown',values.length],['Modes',`${modes.size}/8`],['Fixed γ',a.gamma],['Margin','0.16 m']].map(([k,v])=>`<div class=\"metric\"><span class=\"muted\">${k}</span><b>${v}</b></div>`).join('');audit.innerHTML=`<b>As-built measured bowling scene</b><p>Solid red surfaces are measured physical balls; orange transparent shells are <b>physical radius + 0.16 m</b>. Faint gray surfaces are the old design physical/effective geometry. Dark vertical cylinders are the <b>0.10 m string hard no-go</b>.</p><span class=\"muted\">${cfm?'Safety/balanced are hard-valid successes; reward-dominant intentionally shows two string-safe collision baselines.':'Only hard-valid successes are shown.'} ${g.selection_note} Status S/C/O/T: ${g.status_counts.SUCCESS}/${g.status_counts.COLLISION}/${g.status_counts.OOB}/${g.status_counts.TIMEOUT}.</span>`}else if(group.value==='paper-ready-less-expanded')""",
    )
    return inner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-inner", type=Path, required=True)
    parser.add_argument("--scene-json", type=Path, required=True)
    parser.add_argument("--pre2", type=Path, nargs="+", required=True)
    parser.add_argument("--less-expanded", type=Path, nargs="+", required=True)
    parser.add_argument("--expanded", type=Path, nargs="+", required=True)
    parser.add_argument("--cfmmppi", type=Path, nargs="+", required=True)
    parser.add_argument("--safemppi", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-output", type=Path)
    args = parser.parse_args()

    geometry = load_as_built_geometry(args.scene_json)
    base_inner = args.base_inner.read_text()
    data, data_start, data_end = _extract_data(base_inner)
    goal = data["scene"]["goal"]
    effective = geometry["effective_spheres"]
    z_low = float(np.min(effective[:, 2] - effective[:, 3]) - Z_MARGIN_M)
    z_high = float(np.max(effective[:, 2] + effective[:, 3]) + Z_MARGIN_M)

    policy = {
        "pre2": _dynamic_quality(
            [row for path in args.pre2 for row in _load_policy(path)],
            goal, z_low, z_high,
        ),
        "less-expanded": _dynamic_quality(
            [row for path in args.less_expanded for row in _load_policy(path)],
            goal, z_low, z_high,
        ),
        "expanded": _dynamic_quality(
            [row for path in args.expanded for row in _load_policy(path)],
            goal, z_low, z_high,
        ),
    }
    cfm = _dynamic_quality(
        [
            row
            for path in args.cfmmppi
            for row in torch.load(path, map_location="cpu", weights_only=False)
        ],
        goal, z_low, z_high,
    )
    safe = _dynamic_quality([
        row
        for path in args.safemppi
        for row in torch.load(path, map_location="cpu", weights_only=False)["safemppi"]
    ], goal, z_low, z_high)

    selected = {
        "real-paper-ready-pre2": _route_quota(
            _eligible(policy["pre2"], goal_progress=True),
            {"LLL": 6, "RRR": 2},
        ),
        "real-paper-ready-less-expanded": _balanced(
            _eligible(policy["less-expanded"], goal_progress=True), maximum_modes=4,
        ),
        "real-paper-ready-expanded": _select_seeds(
            _eligible(policy["expanded"], goal_progress=True),
            EXPANDED_SEEDS,
        ),
        "real-paper-ready-cfmmppi": _balanced_cfm(
            cfm,
        ),
        "real-paper-ready-safemppi": _route_quota(
            _eligible(safe, goal_progress=True),
            {"LLL": 2, "LLR": 2, "RRL": 2, "RRR": 2},
        ),
    }
    labels = {
        "real-paper-ready-pre2": "Real PRE2",
        "real-paper-ready-less-expanded": "Real Expanded R1",
        "real-paper-ready-expanded": "Real Expanded S4",
        "real-paper-ready-cfmmppi": "Real CFM-MPPI",
        "real-paper-ready-safemppi": "Real SafeMPPI",
    }
    for name, rows in selected.items():
        viz = []
        for row in rows:
            item = _viz_row(row, labels[name])
            item["id"] = f"{name}:{item['id']}"
            item["hard"] = item["quality"]
            item["samplingTemperature"] = row.get("sampling_temperature")
            viz.append(item)
        data["groups"][name] = viz

    design_effective = np.asarray(data["scene"]["spheres"], float)
    design_physical = design_effective.copy()
    design_physical[:, 3] = 0.1905
    data["realScene"] = {
        "start": data["scene"]["start"],
        "goal": data["scene"]["goal"],
        "bounds": data["scene"]["bounds"],
        "spheres": effective.tolist(),
        "physicalSpheres": geometry["physical_spheres"].tolist(),
        "effectiveSpheres": effective.tolist(),
        "designPhysicalSpheres": design_physical.tolist(),
        "designEffectiveSpheres": design_effective.tolist(),
        "strings": [
            [float(row[0]), float(row[1]), float(row[2] + row[3]), 0.10]
            for row in geometry["physical_spheres"]
        ],
    }
    data["audit"]["real"] = {
        "gamma": GAMMA,
        "effective_margin_m": 0.16,
        "string_radius_m": 0.10,
        "hard_z_band_m": [z_low, z_high],
        "groups": {
            name: {
                "selected": len(rows),
                "selection_note": SELECTION_NOTES[name],
                "modes": sorted({
                    row["stable_route"] for row in rows
                    if row["stable_route"] is not None
                }),
                "seeds": [int(row["rollout_seed"]) for row in rows],
                "sampling_temperatures": [
                    row.get("sampling_temperature") for row in rows
                ],
                "status_counts": {
                    status: int(sum(row["status"] == status for row in rows))
                    for status in ("SUCCESS", "COLLISION", "OOB", "TIMEOUT")
                },
                "all_string_valid": bool(all(
                    (row.get("hard_constraints") or {}).get("string_valid", False)
                    for row in rows
                )),
            }
            for name, rows in selected.items()
        },
    }

    inner = base_inner[:data_start] + json.dumps(
        data, separators=(",", ":"), allow_nan=False,
    ) + base_inner[data_end:]
    options = "".join(
        f'<option value="{name}">{name}</option>' for name in VIEW_NAMES
    )
    marker = "</select>"
    group_start = inner.index('<select id="group">')
    group_end = inner.index(marker, group_start)
    inner = inner[:group_end] + options + inner[group_end:]
    inner = _patch_javascript(inner)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "visualization.inner.html").write_text(inner)
    outer = _outer_document(inner)
    (args.output_dir / "visualization.html").write_text(outer)
    if args.site_output is not None:
        args.site_output.parent.mkdir(parents=True, exist_ok=True)
        args.site_output.write_text(outer)
    (args.output_dir / "real_selection.json").write_text(
        json.dumps(data["audit"]["real"], indent=2) + "\n"
    )
    torch.save({
        "scene": data["realScene"],
        "audit": data["audit"]["real"],
        "groups": selected,
    }, args.output_dir / "real_selected_trajectories.pt")
    print(json.dumps(data["audit"]["real"], indent=2))


if __name__ == "__main__":
    main()
