#!/usr/bin/env python3
"""Render the final native-raw comparison for the speed/exposure experiment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch


DEFAULT_STAGE = Path(
    "/Users/dhl/Documents/safeMPPI_demo_3d/results/"
    "stage1_single_ball_t128/0812_pre2_saved_r1_steps5000_r7"
)
DEFAULT_OUTPUT = Path(
    "/Users/dhl/.codex/visualizations/2026/08/10/"
    "019fe90f-b8eb-7f52-bd16-2bb83e11672e/"
    "speed-exposure-r7-final.html"
)
TASK_CONFIG = Path(
    "/Users/dhl/Documents/safeMPPI_demo_3d/configs/"
    "lab_ball_stage1_goalspace_yminus04_z01_17_r15in_reach03_v1.json"
)
MAX_PATH_POINTS = 12
MAX_HTML_BYTES = 1024 * 1024
FACES = ("x-min", "x-max", "y-min", "y-max", "z-min", "z-max")
ARM_SPECS = (
    {
        "key": "w300_s2500",
        "label": "speed 300 / Adam 2,500",
        "short": "W300 · 2.5k",
        "speed": 300,
        "steps": 2500,
        "folder": "w300_steps2500_eval",
    },
    {
        "key": "w150_s5000",
        "label": "speed 150 / Adam 5,000",
        "short": "W150 · 5k",
        "speed": 150,
        "steps": 5000,
        "folder": "w150_steps5000_eval",
    },
    {
        "key": "w300_s5000",
        "label": "speed 300 / Adam 5,000",
        "short": "W300 · 5k",
        "speed": 300,
        "steps": 5000,
        "folder": "w300_steps5000_eval",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _geometry(task_path: Path) -> dict:
    config = json.loads(task_path.read_text())
    task = config["taskspace"]
    bounds = [
        [float(origin), float(origin + size)]
        for origin, size in zip(task["origin"], task["size"])
    ]
    return {
        "bounds": bounds,
        "start": [float(value) for value in task["start"][:3]],
        "goal": [float(value) for value in task["goal"]],
        "reachRadius": float(task["reach_radius"]),
        "sphere": [
            float(value) for value in config["obstacles"]["spheres"][0]
        ],
    }


def _first_exit(
    dense_path: np.ndarray, bounds: np.ndarray,
) -> tuple[int | None, str | None]:
    outside = np.any(
        (dense_path < bounds[:, 0]) | (dense_path > bounds[:, 1]), axis=1,
    )
    indices = np.flatnonzero(outside)
    if not len(indices):
        return None, None
    index = int(indices[0])
    point = dense_path[index]
    overruns = np.asarray([
        bounds[0, 0] - point[0], point[0] - bounds[0, 1],
        bounds[1, 0] - point[1], point[1] - bounds[1, 1],
        bounds[2, 0] - point[2], point[2] - bounds[2, 1],
    ])
    return index, FACES[int(np.argmax(overruns))]


def _dense_path(row: dict) -> np.ndarray:
    states = np.asarray(row["states"], dtype=np.float64)
    dense_steps = np.asarray(row.get("dense_steps", ()), dtype=np.float64)
    if dense_steps.size:
        return np.concatenate([states[:1, :3], dense_steps.reshape(-1, 3)])
    return states[:, :3]


def _row_group(row: dict, bounds: np.ndarray) -> tuple:
    status = str(row["status"])
    gamma = float(row["gamma"])
    if status == "SUCCESS":
        return gamma, status, str(row.get("mode", "none"))
    if status == "OOB":
        _, face = _first_exit(_dense_path(row), bounds)
        return gamma, status, face or "unknown"
    return gamma, status, "none"


def _representative_rows(rows: list[dict], bounds: np.ndarray) -> list[dict]:
    """Keep two deterministic examples per gamma/outcome/route-or-exit group."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_row_group(row, bounds)].append(row)
    selected = []
    for key in sorted(groups, key=str):
        group = sorted(groups[key], key=lambda row: int(row["episode"]))
        if len(group) == 1:
            selected.extend(group)
        else:
            selected.extend([group[0], group[-1]])
    return selected


def _compact_row(row: dict, bounds: np.ndarray) -> dict:
    dense_path = _dense_path(row)
    exit_index, exit_face = _first_exit(dense_path, bounds)
    indices = set(np.linspace(
        0, len(dense_path) - 1,
        min(MAX_PATH_POINTS, len(dense_path)), dtype=int,
    ).tolist())
    if exit_index is not None:
        indices.update((exit_index, max(0, exit_index - 1)))
    path = np.round(dense_path[sorted(indices)], 3).reshape(-1).tolist()
    exit_payload = None
    if exit_index is not None:
        exit_payload = {
            "f": exit_face,
            "p": np.round(dense_path[exit_index], 4).tolist(),
        }
    return {
        "g": float(row["gamma"]),
        "e": int(row["episode"]),
        "s": str(row["status"]),
        "m": str(row.get("mode", "none")),
        "p": path,
        "x": exit_payload,
        "c": round(float(row["min_clearance_m"]), 4),
        "t": (
            None
            if row.get("time_to_goal_s") is None
            else round(float(row["time_to_goal_s"]), 3)
        ),
    }


def _metric(row: dict) -> dict:
    route_counts = row.get("route_counts", {})
    routes = [
        int(route_counts.get(name, 0))
        for name in ("below", "above", "left", "right")
    ]
    total = sum(routes)
    shares = [value / total if total else 0.0 for value in routes]
    entropy = (
        -sum(value * math.log(value) for value in shares if value) / math.log(4)
        if total else 0.0
    )
    return {
        "n": int(row["episodes"]),
        "sr": round(float(row["SR"]), 6),
        "cr": round(float(row["CR"]), 6),
        "oob": round(float(row["OOB"]), 6),
        "to": round(float(row["timeout"]), 6),
        "v": round(float(row["window_validity"]), 6),
        "cov": int(sum(value > 0 for value in routes)),
        "routes": routes,
        "shares": [round(value, 6) for value in shares],
        "minShare": round(min(shares), 6) if shares else 0.0,
        "entropy": round(entropy, 6),
        "l1": round(sum(abs(value - 0.25) for value in shares), 6),
        "clr": (
            None
            if row.get("successful_min_clearance_m") is None
            else round(float(row["successful_min_clearance_m"]), 6)
        ),
        "ttg": (
            None
            if row.get("successful_time_to_goal_s") is None
            else round(float(row["successful_time_to_goal_s"]), 6)
        ),
    }


def _load_arm(raw_eval: Path, bounds: np.ndarray) -> dict:
    trajectories_path = raw_eval.with_name("raw_trajectories.pt")
    if not trajectories_path.is_file():
        raise FileNotFoundError(trajectories_path)
    evaluation = json.loads(raw_eval.read_text())
    summary = evaluation["summary"]
    trajectories = torch.load(
        trajectories_path, map_location="cpu", weights_only=False,
    )
    rounds = {}
    for round_key in sorted(summary, key=int):
        round_i = int(round_key)
        source_rows = trajectories.get(round_i, trajectories.get(round_key))
        if source_rows is None:
            raise KeyError(f"round {round_i} missing from {trajectories_path}")
        values = summary[round_key]
        representative = _representative_rows(source_rows, bounds)
        rounds[str(round_i)] = {
            "rows": [_compact_row(row, bounds) for row in representative],
            "shown": len(representative),
            "total": len(source_rows),
            "pooled": _metric(values["pooled"]),
            "gamma": {
                str(gamma): _metric(metric)
                for gamma, metric in values["per_gamma"].items()
            },
        }
    return {
        "rounds": rounds,
        "rawEval": str(raw_eval),
        "rawTrajectories": str(trajectories_path),
        "status": evaluation["status"],
    }


def _build_payload(stage: Path, task_path: Path) -> tuple[dict, list[Path]]:
    geometry = _geometry(task_path)
    bounds = np.asarray(geometry["bounds"], dtype=np.float64)
    inputs = [task_path, stage / "QUEUE.json"]
    arms = {}
    for spec in ARM_SPECS:
        raw_eval = stage / "final_inputs" / spec["folder"] / "raw_eval.json"
        loaded = _load_arm(raw_eval, bounds)
        arms[spec["key"]] = {
            "label": spec["label"],
            "short": spec["short"],
            "speed": spec["speed"],
            "steps": spec["steps"],
            **loaded,
        }
        inputs.extend([raw_eval, raw_eval.with_name("raw_trajectories.pt")])
    return {
        **geometry,
        "gammas": ["0.1", "0.3", "0.5", "1"],
        "winner": "w300_s5000",
        "arms": arms,
        "evaluation": {
            "seed": 91000,
            "episodesPerGamma": 40,
            "nfe": 12,
            "nativeRaw": True,
            "expansionSelectorApplied": False,
        },
    }, inputs


FRAGMENT = r'''
<div id="speed-exposure-final">
  <style>
    #speed-exposure-final { color:var(--foreground); font:13px/1.4 ui-sans-serif,system-ui,sans-serif; width:100%; min-width:0; overflow:hidden; }
    #speed-exposure-final * { box-sizing:border-box; }
    #speed-exposure-final .title { font-size:22px; font-weight:720; letter-spacing:-.025em; }
    #speed-exposure-final .sub { color:var(--muted-foreground); margin:3px 0 10px; }
    #speed-exposure-final .verdict { border-left:3px solid var(--viz-series-3); padding:8px 11px; margin:0 0 11px; background:color-mix(in srgb,var(--viz-series-3) 7%,transparent); }
    #speed-exposure-final .verdict strong { font-size:14px; }
    #speed-exposure-final .controls { display:flex; flex-wrap:wrap; gap:8px 14px; align-items:end; padding:9px 0; border-top:1px solid var(--border); border-bottom:1px solid var(--border); }
    #speed-exposure-final .control { display:grid; gap:3px; min-width:112px; }
    #speed-exposure-final label { color:var(--muted-foreground); font-size:10px; font-weight:700; letter-spacing:.045em; text-transform:uppercase; }
    #speed-exposure-final select { color:var(--foreground); background:transparent; border:1px solid var(--border); border-radius:5px; min-height:30px; padding:4px 7px; }
    #speed-exposure-final .kpis { display:grid; grid-template-columns:repeat(8,minmax(70px,1fr)); gap:1px; margin:10px 0 4px; border:1px solid var(--border); border-radius:7px; overflow:hidden; }
    #speed-exposure-final .kpi { padding:7px 8px; min-width:0; border-right:1px solid var(--border); }
    #speed-exposure-final .kpi:last-child { border-right:0; }
    #speed-exposure-final .kpi small { display:block; color:var(--muted-foreground); font-size:10px; text-transform:uppercase; }
    #speed-exposure-final .kpi strong { font-size:16px; white-space:nowrap; }
    #speed-exposure-final .kpi em { display:block; color:var(--muted-foreground); font-size:10px; font-style:normal; white-space:nowrap; }
    #speed-exposure-final .plot3d { width:100%; height:540px; }
    #speed-exposure-final .grid { display:grid; grid-template-columns:1.15fr .85fr; gap:12px; border-top:1px solid var(--border); padding-top:11px; }
    #speed-exposure-final .panel { min-width:0; }
    #speed-exposure-final .panel-head { display:flex; flex-wrap:wrap; justify-content:space-between; gap:7px; align-items:end; min-height:30px; }
    #speed-exposure-final .panel-head strong { font-size:13px; }
    #speed-exposure-final .panel-head select { min-height:27px; font-size:11px; }
    #speed-exposure-final .mini { width:100%; height:300px; }
    #speed-exposure-final .table-wrap { margin-top:12px; overflow-x:auto; border-top:1px solid var(--border); padding-top:10px; }
    #speed-exposure-final table { width:100%; border-collapse:collapse; font-size:12px; }
    #speed-exposure-final th { color:var(--muted-foreground); font-size:10px; text-transform:uppercase; letter-spacing:.035em; text-align:right; padding:5px 7px; border-bottom:1px solid var(--border); }
    #speed-exposure-final th:first-child,#speed-exposure-final td:first-child { text-align:left; }
    #speed-exposure-final td { text-align:right; padding:6px 7px; border-bottom:1px solid color-mix(in srgb,var(--border) 70%,transparent); white-space:nowrap; }
    #speed-exposure-final tr.winner td:first-child { font-weight:750; color:var(--viz-series-3); }
    #speed-exposure-final .foot { color:var(--muted-foreground); font-size:11px; margin-top:8px; }
    @media (max-width:760px) {
      #speed-exposure-final .kpis { grid-template-columns:repeat(4,1fr); }
      #speed-exposure-final .kpi:nth-child(4) { border-right:0; }
      #speed-exposure-final .grid { grid-template-columns:1fr; }
      #speed-exposure-final .plot3d { height:420px; }
    }
    @media (max-width:480px) {
      #speed-exposure-final .controls { display:grid; grid-template-columns:1fr 1fr; }
      #speed-exposure-final .control { min-width:0; }
      #speed-exposure-final select { width:100%; }
      #speed-exposure-final .kpis { grid-template-columns:repeat(2,1fr); }
      #speed-exposure-final .kpi:nth-child(even) { border-right:0; }
    }
  </style>
  <div class="title">Obstacle-speed × optimizer exposure: r0–r7 native-raw result</div>
  <div class="sub">Fixed seed 91000 · 40 episodes/γ · NFE 12 · evaluation uses raw temperature-1 deployment only (no verifier, speed cost, or margin selector)</div>
  <div id="sef-verdict" class="verdict"></div>
  <div class="controls">
    <div class="control"><label for="sef-arm">3D arm</label><select id="sef-arm"></select></div>
    <div class="control"><label for="sef-compare">Overlay</label><select id="sef-compare"><option value="none">none</option></select></div>
    <div class="control"><label for="sef-round">Round</label><select id="sef-round"></select></div>
    <div class="control"><label for="sef-gamma">Gamma</label><select id="sef-gamma"><option value="all">all</option></select></div>
    <div class="control"><label for="sef-outcome">Outcome</label><select id="sef-outcome"><option value="all">all</option><option>SUCCESS</option><option>COLLISION</option><option>OOB</option></select></div>
    <div class="control"><label for="sef-route">Successful route</label><select id="sef-route"><option value="all">all</option><option>below</option><option>above</option><option>left</option><option>right</option></select></div>
  </div>
  <div id="sef-kpis" class="kpis"></div>
  <div id="sef-plot" class="plot3d" role="img" aria-label="Interactive native raw trajectory comparison"></div>
  <div class="grid">
    <div class="panel">
      <div class="panel-head"><strong>All-arm roundwise comparison</strong><select id="sef-trend-metric"><option value="sr">Success rate</option><option value="cr">Collision rate</option><option value="oob">OOB rate</option><option value="v">Window validity</option><option value="clr">Successful clearance</option><option value="ttg">Successful TtG</option><option value="entropy">Mode entropy</option><option value="minShare">Minimum mode share</option></select></div>
      <div id="sef-trend" class="mini" role="img" aria-label="Roundwise metrics for all arms"></div>
    </div>
    <div class="panel">
      <div class="panel-head"><strong>Gamma trend at selected round</strong><select id="sef-gamma-metric"><option value="sr">Success rate</option><option value="cr">Collision rate</option><option value="oob">OOB rate</option><option value="v">Window validity</option><option value="clr">Successful clearance</option><option value="ttg">Successful TtG</option></select></div>
      <div id="sef-gamma-plot" class="mini" role="img" aria-label="Gamma-wise metric comparison"></div>
    </div>
    <div class="panel">
      <div class="panel-head"><strong>Successful route shares at selected round</strong></div>
      <div id="sef-routes" class="mini" role="img" aria-label="Successful route distribution"></div>
    </div>
    <div class="panel">
      <div class="panel-head"><strong>Safety–success frontier across rounds</strong></div>
      <div id="sef-frontier" class="mini" role="img" aria-label="Success rate against collision plus out of bounds rate"></div>
    </div>
  </div>
  <div class="table-wrap"><table><thead><tr><th>r7 arm</th><th>SR</th><th>CR</th><th>OOB</th><th>Validity</th><th>Clearance</th><th>TtG</th><th>Coverage</th><th>b/a/l/r</th><th>Entropy</th><th>Min share</th></tr></thead><tbody id="sef-table"></tbody></table></div>
  <div class="foot">3D paths are a deterministic stratified subset (two examples per γ × outcome × successful route/OOB wall); every metric and table value uses all 160 fixed-bank episodes per round. Expansion-time costs affect sample acquisition only.</div>
  <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
  <script>
  (()=>{
    const root=document.getElementById('speed-exposure-final'),D=__DATA__,q=id=>root.querySelector(`#${id}`);
    const css=getComputedStyle(root),token=n=>css.getPropertyValue(`--viz-series-${n}`).trim();
    const fg=css.getPropertyValue('--foreground').trim(),muted=css.getPropertyValue('--muted-foreground').trim(),border=css.getPropertyValue('--border').trim(),bg=css.getPropertyValue('--background').trim(),destructive=css.getPropertyValue('--destructive').trim();
    const armColors={w300_s2500:token(1),w150_s5000:token(2),w300_s5000:token(3)},modeColors={below:token(1),above:token(2),left:token(3),right:token(4),none:muted},statusColors={COLLISION:destructive,OOB:token(5),TIMEOUT:muted};
    const arm=q('sef-arm'),compare=q('sef-compare'),round=q('sef-round'),gamma=q('sef-gamma'),outcome=q('sef-outcome'),route=q('sef-route'),trendMetric=q('sef-trend-metric'),gammaMetric=q('sef-gamma-metric');
    const armKeys=Object.keys(D.arms),fmt=(v,d=3)=>v==null?'—':Number(v).toFixed(d),pct=v=>`${(100*v).toFixed(1)}%`;
    armKeys.forEach(key=>{for(const el of [arm,compare]){const o=document.createElement('option');o.value=key;o.textContent=D.arms[key].label;el.appendChild(o)}});D.gammas.forEach(g=>{const o=document.createElement('option');o.value=g;o.textContent=g==='1'?'1.0':g;gamma.appendChild(o)});arm.value=D.winner;compare.value='w150_s5000';
    const metric=(key,r,g='all')=>g==='all'?D.arms[key].rounds[r].pooled:D.arms[key].rounds[r].gamma[g];
    function fillRounds(){const old=round.value,ks=Object.keys(D.arms[arm.value].rounds).sort((a,b)=>+a-+b);round.replaceChildren();ks.forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=`r${k}`;round.appendChild(o)});round.value=ks.includes(old)?old:ks.at(-1)}
    const keep=row=>(gamma.value==='all'||String(row.g)===gamma.value)&&(outcome.value==='all'||row.s===outcome.value)&&(route.value==='all'||(row.s==='SUCCESS'&&row.m===route.value));
    function points(row){const z=[];for(let i=0;i<row.p.length;i+=3)z.push([row.p[i],row.p[i+1],row.p[i+2]]);return z}
    function boxTrace(bounds,color,width=2,dash='dot'){const [xb,yb,zb]=bounds,vs=[[xb[0],yb[0],zb[0]],[xb[1],yb[0],zb[0]],[xb[0],yb[1],zb[0]],[xb[1],yb[1],zb[0]],[xb[0],yb[0],zb[1]],[xb[1],yb[0],zb[1]],[xb[0],yb[1],zb[1]],[xb[1],yb[1],zb[1]]],es=[[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]],x=[],y=[],z=[];es.forEach(([a,b])=>{for(const i of [a,b]){x.push(vs[i][0]);y.push(vs[i][1]);z.push(vs[i][2])}x.push(null);y.push(null);z.push(null)});return{type:'scatter3d',mode:'lines',x,y,z,line:{color,width,dash},hoverinfo:'skip',showlegend:false}}
    function sphereTrace(){const [cx,cy,cz,r]=D.sphere,x=[],y=[],z=[],i=[],j=[],k=[],nu=22,nv=13;for(let b=0;b<nv;b++){const p=Math.PI*b/(nv-1);for(let a=0;a<nu;a++){const t=2*Math.PI*a/nu;x.push(cx+r*Math.sin(p)*Math.cos(t));y.push(cy+r*Math.sin(p)*Math.sin(t));z.push(cz+r*Math.cos(p))}}for(let b=0;b<nv-1;b++)for(let a=0;a<nu;a++){const u=b*nu+a,v=b*nu+(a+1)%nu,w=(b+1)*nu+a,n=(b+1)*nu+(a+1)%nu;i.push(u,v);j.push(w,w);k.push(v,n)}return{type:'mesh3d',x,y,z,i,j,k,color:muted,opacity:.28,hoverinfo:'skip',name:'sphere',showlegend:true}}
    function joined(rows){const x=[],y=[],z=[],text=[];rows.forEach(row=>{const label=`γ=${row.g} · ${row.s} · ${row.m} · ep${row.e}${row.x?` · first exit ${row.x.f}`:''}`;points(row).forEach(p=>{x.push(p[0]);y.push(p[1]);z.push(p[2]);text.push(label)});x.push(null);y.push(null);z.push(null);text.push('')});return{x,y,z,text}}
    function pathTrace(rows,name,color,width=3,dash='solid',opacity=.62){return{type:'scatter3d',mode:'lines',...joined(rows),line:{color,width,dash},opacity,name,hovertemplate:'%{text}<extra></extra>',showlegend:true}}
    function delta(v,r,key){if(!r)return'';const d=v-r;return `<em>Δ overlay ${d>=0?'+':''}${fmt(d)}</em>`}
    function draw3d(){const selected=D.arms[arm.value],rows=selected.rounds[round.value].rows.filter(keep),traces=[boxTrace(D.bounds,border),sphereTrace()];if(compare.value!=='none'&&compare.value!==arm.value){const other=D.arms[compare.value].rounds[round.value].rows.filter(keep);if(other.length)traces.push(pathTrace(other,`${D.arms[compare.value].short} overlay`,muted,2,'dot',.22))}const groups=new Map();rows.forEach(row=>{const key=row.s==='SUCCESS'?`SUCCESS · ${row.m}`:row.s;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(row)});groups.forEach((values,key)=>{const row=values[0],color=row.s==='SUCCESS'?modeColors[row.m]:statusColors[row.s];traces.push(pathTrace(values,key,color,row.s==='OOB'?4:3,'solid',row.s==='OOB'?.9:.65))});const exits=rows.filter(row=>row.x);if(exits.length)traces.push({type:'scatter3d',mode:'markers',x:exits.map(row=>row.x.p[0]),y:exits.map(row=>row.x.p[1]),z:exits.map(row=>row.x.p[2]),text:exits.map(row=>`γ=${row.g} · ep${row.e} · ${row.x.f}`),hovertemplate:'first OOB %{text}<extra></extra>',marker:{size:5,color:token(5),symbol:'x'},name:'first OOB exit'});traces.push({type:'scatter3d',mode:'lines+markers',x:[D.start[0],D.goal[0]],y:[D.start[1],D.goal[1]],z:[D.start[2],D.goal[2]],line:{color:fg,width:2},marker:{size:[5,7],color:[fg,token(6)],symbol:['square','diamond']},text:['start','goal'],hovertemplate:'%{text}<extra></extra>',name:'start–goal'});Plotly.react(q('sef-plot'),traces,{margin:{l:0,r:0,t:6,b:0},paper_bgcolor:'rgba(0,0,0,0)',font:{color:fg,size:11},legend:{orientation:'h',y:1.02,x:0,bgcolor:'rgba(0,0,0,0)'},scene:{aspectmode:'data',xaxis:{title:'x [m]',range:D.bounds[0],backgroundcolor:bg,gridcolor:border},yaxis:{title:'y [m]',range:D.bounds[1],backgroundcolor:bg,gridcolor:border},zaxis:{title:'z [m]',range:D.bounds[2],backgroundcolor:bg,gridcolor:border},camera:{eye:{x:1.45,y:1.45,z:.9}}}},{responsive:true,displaylogo:false});const m=metric(arm.value,round.value,gamma.value),ref=compare.value==='none'?null:metric(compare.value,round.value,gamma.value);const kpis=[['SR',m.sr,ref&&ref.sr],['CR',m.cr,ref&&ref.cr],['OOB',m.oob,ref&&ref.oob],['validity',m.v,ref&&ref.v],['clearance',m.clr,ref&&ref.clr,' m'],['TtG',m.ttg,ref&&ref.ttg,' s'],['coverage',`${m.cov}/4`,null],['mode entropy',m.entropy,ref&&ref.entropy]];q('sef-kpis').innerHTML=kpis.map(([name,v,r,suffix=''])=>`<div class="kpi"><small>${name}</small><strong>${typeof v==='string'?v:fmt(v)}${suffix}</strong>${typeof v==='number'?delta(v,r,name):''}</div>`).join('')}
    const axisTitle={sr:'success rate',cr:'collision rate',oob:'OOB rate',v:'window validity',clr:'clearance [m]',ttg:'time to goal [s]',entropy:'normalized entropy',minShare:'minimum mode share'};
    function drawTrend(){const key=trendMetric.value,traces=armKeys.map(a=>{const ks=Object.keys(D.arms[a].rounds).sort((x,y)=>+x-+y);return{type:'scatter',mode:'lines+markers',name:D.arms[a].short,x:ks.map(Number),y:ks.map(r=>metric(a,r)[key]),line:{color:armColors[a],width:a===D.winner?3:2},marker:{size:a===D.winner?7:5},hovertemplate:`r%{x}<br>${axisTitle[key]}: %{y:.3f}<extra>${D.arms[a].short}</extra>`}});const rate=['sr','cr','oob','v','entropy','minShare'].includes(key);Plotly.react(q('sef-trend'),traces,{margin:{l:54,r:12,t:14,b:45},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:fg,size:11},xaxis:{title:'round',dtick:1,gridcolor:border},yaxis:{title:axisTitle[key],range:rate?[0,1]:undefined,gridcolor:border},legend:{orientation:'h',y:1.12,x:0,bgcolor:'rgba(0,0,0,0)'}},{responsive:true,displaylogo:false})}
    function drawGamma(){const key=gammaMetric.value,x=D.gammas.map(Number),traces=armKeys.map(a=>({type:'scatter',mode:'lines+markers',name:D.arms[a].short,x,y:D.gammas.map(g=>metric(a,round.value,g)[key]),line:{color:armColors[a],width:a===D.winner?3:2},marker:{size:a===D.winner?7:5},hovertemplate:`γ=%{x}<br>${axisTitle[key]}: %{y:.3f}<extra>${D.arms[a].short}</extra>`}));const rate=['sr','cr','oob','v'].includes(key);Plotly.react(q('sef-gamma-plot'),traces,{margin:{l:54,r:12,t:14,b:45},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:fg,size:11},xaxis:{title:'gamma',tickvals:x,gridcolor:border},yaxis:{title:axisTitle[key],range:rate?[0,1]:undefined,gridcolor:border},legend:{orientation:'h',y:1.12,x:0,bgcolor:'rgba(0,0,0,0)'}},{responsive:true,displaylogo:false})}
    function drawRoutes(){const modes=['below','above','left','right'],traces=modes.map((mode,i)=>({type:'bar',name:mode,x:armKeys.map(a=>D.arms[a].short),y:armKeys.map(a=>metric(a,round.value).shares[i]),marker:{color:modeColors[mode]},hovertemplate:`${mode}: %{y:.1%}<extra></extra>`}));Plotly.react(q('sef-routes'),traces,{barmode:'stack',margin:{l:52,r:12,t:14,b:48},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:fg,size:11},xaxis:{gridcolor:border},yaxis:{title:'share among successes',range:[0,1],tickformat:'.0%',gridcolor:border},shapes:[{type:'line',x0:-.5,x1:2.5,y0:.25,y1:.25,line:{color:fg,width:1,dash:'dash'}}],legend:{orientation:'h',y:1.12,x:0,bgcolor:'rgba(0,0,0,0)'}},{responsive:true,displaylogo:false})}
    function drawFrontier(){const traces=armKeys.map(a=>{const ks=Object.keys(D.arms[a].rounds).sort((x,y)=>+x-+y),ms=ks.map(r=>metric(a,r));return{type:'scatter',mode:'lines+markers+text',name:D.arms[a].short,x:ms.map(m=>m.cr+m.oob+m.to),y:ms.map(m=>m.sr),text:ks.map(r=>`r${r}`),textposition:'top center',textfont:{size:9},line:{color:armColors[a],width:a===D.winner?3:2},marker:{size:a===D.winner?8:6},hovertemplate:'failure=%{x:.3f}<br>SR=%{y:.3f}<br>%{text}<extra>'+D.arms[a].short+'</extra>'}});Plotly.react(q('sef-frontier'),traces,{margin:{l:54,r:12,t:14,b:48},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:fg,size:11},xaxis:{title:'CR + OOB + timeout (lower is better)',range:[0,1],gridcolor:border},yaxis:{title:'success rate (higher is better)',range:[0,1],gridcolor:border},legend:{orientation:'h',y:1.12,x:0,bgcolor:'rgba(0,0,0,0)'}},{responsive:true,displaylogo:false})}
    function fillTable(){q('sef-table').innerHTML=armKeys.map(a=>{const m=metric(a,'7');return `<tr class="${a===D.winner?'winner':''}"><td>${D.arms[a].label}${a===D.winner?' ★':''}</td><td>${fmt(m.sr)}</td><td>${fmt(m.cr)}</td><td>${fmt(m.oob)}</td><td>${fmt(m.v)}</td><td>${fmt(m.clr)} m</td><td>${fmt(m.ttg,2)} s</td><td>${m.cov}/4</td><td>${m.routes.join('/')}</td><td>${fmt(m.entropy)}</td><td>${pct(m.minShare)}</td></tr>`}).join('')}
    function verdict(){const w=metric(D.winner,'7'),base=metric('w300_s2500','7'),alt=metric('w150_s5000','7');q('sef-verdict').innerHTML=`<strong>Winner: ${D.arms[D.winner].label} at r7</strong> — SR ${fmt(w.sr)}, CR ${fmt(w.cr)}, OOB ${fmt(w.oob)}, validity ${fmt(w.v)}, coverage ${w.cov}/4. Doubling Adam exposure at W300 recovered all four modes and changed r7 by <strong>ΔSR +${fmt(w.sr-base.sr)}, ΔCR ${fmt(w.cr-base.cr)}, ΔOOB ${fmt(w.oob-base.oob)}</strong> versus 2,500 steps. It also beats W150/5k on SR by +${fmt(w.sr-alt.sr)} and OOB by ${fmt(w.oob-alt.oob)}, but CR=.150 and OOB=.063 mean this is the nearest winner—not yet paper-ready.`}
    function draw(){draw3d();drawTrend();drawGamma();drawRoutes();drawFrontier()}
    arm.addEventListener('change',()=>{fillRounds();draw()});[compare,round,gamma,outcome,route].forEach(el=>el.addEventListener('change',draw));trendMetric.addEventListener('change',drawTrend);gammaMetric.addEventListener('change',drawGamma);fillRounds();verdict();fillTable();draw();
  })();
  </script>
</div>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--task-config", type=Path, default=TASK_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload, inputs = _build_payload(args.stage, args.task_config)
    data = json.dumps(
        payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    )
    fragment = FRAGMENT.replace("__DATA__", data)
    if re.search(r"<(?:!doctype|html|head|body)\b", fragment, flags=re.I):
        raise ValueError("visualization must remain an HTML fragment")
    size = len(fragment.encode())
    if size >= MAX_HTML_BYTES:
        raise ValueError(f"fragment is too large: {size} bytes")
    _atomic_write(args.output, fragment)

    provenance = {
        "kind": "PRE2 obstacle-speed by optimizer-exposure native-raw final report",
        "generated_unix": time.time(),
        "stage": str(args.stage),
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": _sha256(args.output),
        "winner": payload["winner"],
        "inputs": {str(path): _sha256(path) for path in dict.fromkeys(inputs)},
        "trajectory_display": (
            "deterministic two-per-stratum subset; metrics use all 160 episodes/round"
        ),
    }
    provenance_path = args.output.with_suffix(".provenance.json")
    _atomic_write(
        provenance_path,
        json.dumps(provenance, indent=2, allow_nan=False) + "\n",
    )
    print(
        f"wrote {args.output} ({args.output.stat().st_size} bytes); "
        f"winner={payload['winner']}"
    )


if __name__ == "__main__":
    main()
