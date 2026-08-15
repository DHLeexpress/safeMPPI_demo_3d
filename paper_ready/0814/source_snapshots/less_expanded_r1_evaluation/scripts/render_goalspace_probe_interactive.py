#!/usr/bin/env python3
"""Render old/new task-space detours and OOB exits from fixed-bank rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch


MODES = ("below", "above", "left", "right")
FACES = ("x-min", "x-max", "y-min", "y-max", "z-min", "z-max")
MAX_POINTS = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _arm(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("arm must be LABEL=EVAL_DIR")
    label, raw_path = text.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("arm must be LABEL=EVAL_DIR")
    return label, Path(raw_path)


def _reference(text: str) -> tuple[str, int]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("reference must be LABEL:ROUND")
    label, round_text = text.rsplit(":", 1)
    return label, int(round_text)


def _parse_bounds(text: str) -> np.ndarray:
    values = [float(value) for value in text.split(",")]
    if len(values) != 6:
        raise argparse.ArgumentTypeError(
            "bounds must be xmin,xmax,ymin,ymax,zmin,zmax"
        )
    return np.asarray(values, np.float64).reshape(3, 2)


def _dense_path(row: dict) -> np.ndarray:
    states = np.asarray(row["states"], np.float64)
    dense = np.asarray(row.get("dense_steps", ()), np.float64)
    if dense.size:
        return np.concatenate([states[:1, :3], dense.reshape(-1, 3)])
    return states[:, :3]


def _first_exit(path: np.ndarray, bounds: np.ndarray) -> tuple[int | None, str | None]:
    outside = np.any(
        (path < bounds[:, 0] - 1e-7)
        | (path > bounds[:, 1] + 1e-7),
        axis=1,
    )
    indices = np.flatnonzero(outside)
    if not len(indices):
        return None, None
    index = int(indices[0])
    point = path[index]
    overrun = np.asarray([
        bounds[0, 0] - point[0], point[0] - bounds[0, 1],
        bounds[1, 0] - point[1], point[1] - bounds[1, 1],
        bounds[2, 0] - point[2], point[2] - bounds[2, 1],
    ])
    return index, FACES[int(np.argmax(overrun))]


def _compact(row: dict, old_bounds: np.ndarray, new_bounds: np.ndarray) -> dict:
    path = _dense_path(row)
    old_y = float(old_bounds[1, 0])
    old_y_indices = np.flatnonzero(path[:, 1] < old_y - 1e-7)
    old_index = int(old_y_indices[0]) if len(old_y_indices) else None
    new_exit_index, new_exit_face = _first_exit(path, new_bounds)
    turn_index = None
    returned = False
    if old_index is not None:
        later = path[old_index + 1:]
        return_indices = np.flatnonzero(later[:, 1] >= old_y - 1e-7)
        returned = bool(len(return_indices))
        delta_y = np.diff(path[:, 1])
        reversals = np.flatnonzero(delta_y[old_index:] > 1e-7)
        if len(reversals):
            turn_index = old_index + int(reversals[0]) + 1

    indices = set(np.linspace(
        0, len(path) - 1, min(MAX_POINTS, len(path)), dtype=int,
    ).tolist())
    for event_index in (old_index, turn_index, new_exit_index):
        if event_index is not None:
            indices.update({max(0, event_index - 1), event_index})
    compact_path = np.round(path[sorted(indices)], 3).reshape(-1).tolist()
    return {
        "g": float(row["gamma"]),
        "e": int(row["episode"]),
        "s": str(row["status"]),
        "m": str(row.get("mode", "none")),
        "p": compact_path,
        "old": (
            None if old_index is None
            else np.round(path[old_index], 4).tolist()
        ),
        "turn": (
            None if turn_index is None
            else np.round(path[turn_index], 4).tolist()
        ),
        "returned": returned,
        "newExit": (
            None if new_exit_index is None else {
                "face": new_exit_face,
                "point": np.round(path[new_exit_index], 4).tolist(),
            }
        ),
    }


def _metric(values: dict) -> dict:
    routes = values.get("route_counts", {})
    return {
        "n": int(values["episodes"]),
        "sr": float(values["SR"]),
        "cr": float(values["CR"]),
        "oob": float(values["OOB"]),
        "timeout": float(values["timeout"]),
        "validity": float(values["window_validity"]),
        "clearance": values.get("successful_min_clearance_m"),
        "ttg": values.get("successful_time_to_goal_s"),
        "coverage": float(values.get("route_coverage", 0.0)),
        "routes": [int(routes.get(mode, 0)) for mode in MODES],
    }


def _load_arm(
    eval_dir: Path,
    old_bounds: np.ndarray,
    new_bounds: np.ndarray,
    requested_rounds: set[int] | None,
) -> tuple[dict, list[Path]]:
    raw_eval = eval_dir / "raw_eval.json"
    trajectories_path = eval_dir / "raw_trajectories.pt"
    if not raw_eval.is_file() or not trajectories_path.is_file():
        raise FileNotFoundError(f"evaluation is not fully published: {eval_dir}")
    evaluation = json.loads(raw_eval.read_text())
    trajectories = torch.load(
        trajectories_path, map_location="cpu", weights_only=False,
    )
    rounds = {}
    for round_text in sorted(evaluation["summary"], key=int):
        round_i = int(round_text)
        if requested_rounds is not None and round_i not in requested_rounds:
            continue
        rows = trajectories.get(round_i, trajectories.get(round_text))
        if rows is None:
            raise KeyError(f"round {round_i} missing from {trajectories_path}")
        summary = evaluation["summary"][round_text]
        rounds[round_text] = {
            "rows": [_compact(row, old_bounds, new_bounds) for row in rows],
            "pooled": _metric(summary["pooled"]),
            "gamma": {
                f"{float(gamma):g}": _metric(values)
                for gamma, values in summary["per_gamma"].items()
            },
        }
    if not rounds:
        raise ValueError(f"no requested rounds in {raw_eval}")
    return {"rounds": rounds}, [raw_eval, trajectories_path]


def _geometry(task: dict) -> dict:
    taskspace = task["taskspace"]
    return {
        "bounds": [
            [float(origin), float(origin + size)]
            for origin, size in zip(taskspace["origin"], taskspace["size"])
        ],
        "start": [float(value) for value in taskspace["start"][:3]],
        "goal": [float(value) for value in taskspace["goal"]],
        "sphere": [
            float(value) for value in task["obstacles"]["spheres"][0]
        ],
    }


FRAGMENT = r'''
<div id="goalspace-detour-oob-lab">
  <style>
    #goalspace-detour-oob-lab { color: var(--foreground); font: 13px/1.35 ui-sans-serif, system-ui, sans-serif; width: 100%; min-width: 0; overflow: hidden; }
    #goalspace-detour-oob-lab * { box-sizing: border-box; }
    #goalspace-detour-oob-lab .title { font-size: 20px; font-weight: 500; margin-bottom: 3px; }
    #goalspace-detour-oob-lab .sub { color: var(--muted-foreground); margin-bottom: 9px; }
    #goalspace-detour-oob-lab .controls { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: end; padding: 8px 0 10px; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
    #goalspace-detour-oob-lab .control { display: grid; gap: 3px; min-width: 102px; }
    #goalspace-detour-oob-lab label { color: var(--muted-foreground); font-size: 11px; font-weight: 500; text-transform: uppercase; }
    #goalspace-detour-oob-lab select { color: var(--foreground); background: transparent; border: 1px solid var(--border); border-radius: 5px; min-height: 30px; padding: 4px 7px; }
    #goalspace-detour-oob-lab .check { display: flex; gap: 6px; align-items: center; min-height: 30px; color: var(--foreground); font-size: 12px; text-transform: none; }
    #goalspace-detour-oob-lab .summary { display: flex; flex-wrap: wrap; gap: 6px 16px; padding: 9px 0 2px; min-height: 42px; }
    #goalspace-detour-oob-lab .summary span { white-space: nowrap; }
    #goalspace-detour-oob-lab .plot3d { width: 100%; height: 555px; }
    #goalspace-detour-oob-lab .mini-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; border-top: 1px solid var(--border); padding-top: 10px; }
    #goalspace-detour-oob-lab .mini { min-width: 0; height: 280px; }
    #goalspace-detour-oob-lab table { width: 100%; border-collapse: collapse; margin-top: 8px; font-variant-numeric: tabular-nums; }
    #goalspace-detour-oob-lab th, #goalspace-detour-oob-lab td { padding: 4px 7px; border-bottom: 1px solid var(--border); text-align: right; }
    #goalspace-detour-oob-lab th:first-child, #goalspace-detour-oob-lab td:first-child { text-align: left; }
    @media (max-width: 620px) {
      #goalspace-detour-oob-lab .controls { display: grid; grid-template-columns: 1fr 1fr; }
      #goalspace-detour-oob-lab .control { min-width: 0; }
      #goalspace-detour-oob-lab select { width: 100%; }
      #goalspace-detour-oob-lab .plot3d { height: 420px; }
      #goalspace-detour-oob-lab .mini-grid { grid-template-columns: 1fr; }
    }
  </style>
  <div class="title">Expanded goal-space: detours and outer-boundary exits</div>
  <div class="sub">Fixed seed 91000 · NFE 12 · 40 episodes/γ · old y-min −1.7 m versus new y-min −2.1 m</div>
  <div class="controls">
    <div class="control"><label for="gs-arm">Arm</label><select id="gs-arm"></select></div>
    <div class="control"><label for="gs-round">Round</label><select id="gs-round"></select></div>
    <div class="control"><label for="gs-gamma">Gamma</label><select id="gs-gamma"><option value="all">all</option></select></div>
    <div class="control"><label for="gs-outcome">Outcome</label><select id="gs-outcome"><option value="all">all</option><option>SUCCESS</option><option>COLLISION</option><option>OOB</option><option>TIMEOUT</option></select></div>
    <div class="control"><label for="gs-mode">Mode</label><select id="gs-mode"><option value="all">all</option><option>below</option><option>above</option><option>left</option><option>right</option><option>none</option></select></div>
    <div class="control"><label for="gs-detour">Old-wall behavior</label><select id="gs-detour"><option value="all">all</option><option value="cross">crossed old y-min</option><option value="return">crossed then returned</option><option value="outer">new-boundary exit</option></select></div>
    <label class="check"><input id="gs-overlay" type="checkbox" checked> overlay reference</label>
  </div>
  <div id="gs-summary" class="summary" aria-live="polite"></div>
  <div id="gs-plot" class="plot3d" role="img" aria-label="Interactive 3D trajectories with old-wall crossings, turnarounds, and new-boundary exits"></div>
  <div class="mini-grid"><div id="gs-trend" class="mini"></div><div id="gs-boundary" class="mini"></div></div>
  <table aria-label="Gamma-wise fixed-bank metrics"><thead><tr><th>γ</th><th>SR</th><th>CR</th><th>OOB</th><th>validity</th><th>clearance m</th><th>TtG s</th><th>b/a/l/r</th></tr></thead><tbody id="gs-gamma-table"></tbody></table>
  <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
  <script>
  (()=>{
    const root=document.getElementById('goalspace-detour-oob-lab'),D=__DATA__,q=id=>root.querySelector(`#${id}`);
    const css=getComputedStyle(root),token=n=>css.getPropertyValue(`--viz-series-${n}`).trim(),fg=css.getPropertyValue('--foreground').trim(),muted=css.getPropertyValue('--muted-foreground').trim(),border=css.getPropertyValue('--border').trim(),bg=css.getPropertyValue('--background').trim();
    const modeColor={below:token(1),above:token(2),left:token(3),right:token(4),none:muted},statusColor={COLLISION:css.getPropertyValue('--destructive').trim(),OOB:token(5),TIMEOUT:muted};
    const arm=q('gs-arm'),round=q('gs-round'),gamma=q('gs-gamma'),outcome=q('gs-outcome'),mode=q('gs-mode'),detour=q('gs-detour'),overlay=q('gs-overlay');
    Object.keys(D.arms).forEach(key=>{const o=document.createElement('option');o.value=key;o.textContent=key;arm.appendChild(o)});D.gammas.forEach(value=>{const o=document.createElement('option');o.value=value;o.textContent=value==='1'?'1.0':value;gamma.appendChild(o)});
    arm.value=Object.keys(D.arms).find(key=>key!==D.reference.arm)||Object.keys(D.arms)[0];
    const fmt=(v,d=3)=>v==null?'—':Number(v).toFixed(d),points=row=>{const p=[];for(let i=0;i<row.p.length;i+=3)p.push([row.p[i],row.p[i+1],row.p[i+2]]);return p};
    function fillRounds(){const prior=round.value,keys=Object.keys(D.arms[arm.value].rounds).sort((a,b)=>+a-+b);round.replaceChildren();keys.forEach(key=>{const o=document.createElement('option');o.value=key;o.textContent=`r${key}`;round.appendChild(o)});round.value=keys.includes(prior)?prior:keys.at(-1)}
    function boxTrace(bounds,name,color,dash,width){const [xb,yb,zb]=bounds,vs=[[xb[0],yb[0],zb[0]],[xb[1],yb[0],zb[0]],[xb[0],yb[1],zb[0]],[xb[1],yb[1],zb[0]],[xb[0],yb[0],zb[1]],[xb[1],yb[0],zb[1]],[xb[0],yb[1],zb[1]],[xb[1],yb[1],zb[1]]],es=[[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]],x=[],y=[],z=[];es.forEach(([a,b])=>{[a,b].forEach(i=>{x.push(vs[i][0]);y.push(vs[i][1]);z.push(vs[i][2])});x.push(null);y.push(null);z.push(null)});return{type:'scatter3d',mode:'lines',x,y,z,line:{color,dash,width},hoverinfo:'skip',name,showlegend:true}}
    function sphereTrace(){const [cx,cy,cz,r]=D.sphere,x=[],y=[],z=[],i=[],j=[],k=[],nu=22,nv=13;for(let b=0;b<nv;b++){const p=Math.PI*b/(nv-1);for(let a=0;a<nu;a++){const t=2*Math.PI*a/nu;x.push(cx+r*Math.sin(p)*Math.cos(t));y.push(cy+r*Math.sin(p)*Math.sin(t));z.push(cz+r*Math.cos(p))}}for(let b=0;b<nv-1;b++)for(let a=0;a<nu;a++){const u=b*nu+a,v=b*nu+(a+1)%nu,w=(b+1)*nu+a,n=(b+1)*nu+(a+1)%nu;i.push(u,v);j.push(w,w);k.push(v,n)}return{type:'mesh3d',x,y,z,i,j,k,color:muted,opacity:.25,hoverinfo:'skip',name:'sphere'}}
    function keep(row){const behavior=detour.value==='all'||(detour.value==='cross'&&row.old)||(detour.value==='return'&&row.returned)||(detour.value==='outer'&&row.newExit);return(gamma.value==='all'||String(row.g)===gamma.value)&&(outcome.value==='all'||row.s===outcome.value)&&(mode.value==='all'||row.m===mode.value)&&behavior}
    function joined(rows){const x=[],y=[],z=[],text=[];rows.forEach(row=>{const label=`γ=${row.g} · ${row.s} · ${row.m} · ep${row.e}`;points(row).forEach(p=>{x.push(p[0]);y.push(p[1]);z.push(p[2]);text.push(label)});x.push(null);y.push(null);z.push(null);text.push('')});return{x,y,z,text}}
    function pathTrace(rows,name,color,width=3,dash='solid',opacity=.6){return{type:'scatter3d',mode:'lines',...joined(rows),line:{color,width,dash},opacity,name,hovertemplate:'%{text}<extra></extra>'}}
    function currentMetric(data){return gamma.value==='all'?data.rounds[round.value].pooled:data.rounds[round.value].gamma[gamma.value]}
    function draw3d(){const data=D.arms[arm.value],rows=data.rounds[round.value].rows.filter(keep),traces=[boxTrace(D.oldBounds,'old taskspace',token(6),'dash',3),boxTrace(D.newBounds,'new taskspace',border,'dot',3),sphereTrace()];
      if(overlay.checked&&(arm.value!==D.reference.arm||+round.value!==D.reference.round)){const ref=D.arms[D.reference.arm].rounds[String(D.reference.round)];if(ref){const refRows=ref.rows.filter(keep);if(refRows.length)traces.push(pathTrace(refRows,'reference',muted,2,'dot',.18))}}
      const groups=new Map();rows.forEach(row=>{const key=row.s==='SUCCESS'?`SUCCESS · ${row.m}`:row.s;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(row)});groups.forEach((values,key)=>{const row=values[0],color=row.s==='SUCCESS'?modeColor[row.m]:statusColor[row.s];traces.push(pathTrace(values,key,color,row.s==='OOB'?4:3,'solid',row.s==='OOB'?.82:.58))});
      const crossed=rows.filter(row=>row.old),turned=rows.filter(row=>row.turn),exits=rows.filter(row=>row.newExit);if(crossed.length)traces.push({type:'scatter3d',mode:'markers',x:crossed.map(r=>r.old[0]),y:crossed.map(r=>r.old[1]),z:crossed.map(r=>r.old[2]),marker:{size:4,color:token(6),symbol:'circle-open'},text:crossed.map(r=>`old y-min crossed · γ=${r.g} · ep${r.e}`),hovertemplate:'%{text}<extra></extra>',name:'old-wall crossing'});if(turned.length)traces.push({type:'scatter3d',mode:'markers',x:turned.map(r=>r.turn[0]),y:turned.map(r=>r.turn[1]),z:turned.map(r=>r.turn[2]),marker:{size:5,color:token(3),symbol:'diamond-open'},text:turned.map(r=>`turnaround · γ=${r.g} · ep${r.e} · returned=${r.returned}`),hovertemplate:'%{text}<extra></extra>',name:'turnaround'});if(exits.length)traces.push({type:'scatter3d',mode:'markers',x:exits.map(r=>r.newExit.point[0]),y:exits.map(r=>r.newExit.point[1]),z:exits.map(r=>r.newExit.point[2]),marker:{size:5,color:token(5),symbol:'x'},text:exits.map(r=>`new exit ${r.newExit.face} · γ=${r.g} · ep${r.e}`),hovertemplate:'%{text}<extra></extra>',name:'new-boundary exit'});
      traces.push({type:'scatter3d',mode:'lines+markers',x:[D.start[0],D.goal[0]],y:[D.start[1],D.goal[1]],z:[D.start[2],D.goal[2]],line:{color:fg,width:2},marker:{size:[5,7],color:[fg,token(6)],symbol:['square','diamond']},text:['start','goal'],hovertemplate:'%{text}<extra></extra>',name:'start–goal'});
      Plotly.react(q('gs-plot'),traces,{margin:{l:0,r:0,t:8,b:0},paper_bgcolor:'rgba(0,0,0,0)',font:{color:fg,size:11},legend:{orientation:'h',y:1.02,x:0,bgcolor:'rgba(0,0,0,0)'},scene:{aspectmode:'data',xaxis:{title:'x [m]',range:D.newBounds[0],backgroundcolor:bg,gridcolor:border},yaxis:{title:'y [m]',range:D.newBounds[1],backgroundcolor:bg,gridcolor:border},zaxis:{title:'z [m]',range:D.newBounds[2],backgroundcolor:bg,gridcolor:border},camera:{eye:{x:1.45,y:1.45,z:.9}}}},{responsive:true,displaylogo:false});
      const metric=currentMetric(data),cross=rows.filter(r=>r.old).length,returned=rows.filter(r=>r.returned).length,newExit=rows.filter(r=>r.newExit).length,xmax=rows.filter(r=>r.newExit&&r.newExit.face==='x-max').length,ymin=rows.filter(r=>r.newExit&&r.newExit.face==='y-min').length,lr=metric.routes[2]+metric.routes[3],lrImbalance=lr?Math.abs(metric.routes[3]-metric.routes[2])/lr:null;q('gs-summary').innerHTML=`<span>shown <strong>${rows.length}/${data.rounds[round.value].rows.length}</strong></span><span>SR <strong>${fmt(metric.sr)}</strong></span><span>CR <strong>${fmt(metric.cr)}</strong></span><span>OOB <strong>${fmt(metric.oob)}</strong></span><span>validity <strong>${fmt(metric.validity)}</strong></span><span>clearance <strong>${fmt(metric.clearance)} m</strong></span><span>TtG <strong>${fmt(metric.ttg,2)} s</strong></span><span>routes <strong>${metric.routes.join('/')}</strong></span><span>L/R imbalance <strong>${fmt(lrImbalance)}</strong></span><span>old-y cross / return / new exit <strong>${cross}/${returned}/${newExit}</strong></span><span>new x-max / y-min exits <strong>${xmax}/${ymin}</strong></span>`;
    }
    function drawMinis(){const data=D.arms[arm.value],keys=Object.keys(data.rounds).sort((a,b)=>+a-+b),x=keys.map(Number),pooled=keys.map(key=>data.rounds[key].pooled),series=[['SR','sr',token(3)],['CR','cr',token(2)],['OOB','oob',token(5)],['validity','validity',token(4)]];Plotly.react(q('gs-trend'),series.map(([name,key,color])=>({type:'scatter',mode:'lines+markers',name,x,y:pooled.map(row=>row[key]),line:{color,width:2},marker:{size:5},hovertemplate:`r%{x}<br>${name}: %{y:.3f}<extra></extra>`})),{title:{text:'Fixed-bank rates',x:0,font:{size:14}},margin:{l:48,r:12,t:58,b:42},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:fg,size:11},xaxis:{title:'round',gridcolor:border,dtick:1},yaxis:{title:'rate',range:[0,1],gridcolor:border},legend:{orientation:'h',y:1.14,x:.15,bgcolor:'rgba(0,0,0,0)'}},{responsive:true,displaylogo:false});
      const count=(key,predicate)=>data.rounds[key].rows.filter(predicate).length,boundary=[['old-y cross',r=>r.old,token(6)],['returned',r=>r.returned,token(3)],['success after cross',r=>r.old&&r.s==='SUCCESS',token(4)],['new exit',r=>r.newExit,token(5)]];Plotly.react(q('gs-boundary'),boundary.map(([name,test,color])=>({type:'bar',name,x,y:keys.map(key=>count(key,test)),marker:{color},hovertemplate:`r%{x}<br>${name}: %{y}<extra></extra>`})),{barmode:'group',title:{text:'Old-wall detour versus new exits',x:0,font:{size:14}},margin:{l:48,r:12,t:58,b:42},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:fg,size:11},xaxis:{title:'round',gridcolor:border,dtick:1},yaxis:{title:'episodes',rangemode:'tozero',gridcolor:border},legend:{orientation:'h',y:1.17,x:0,bgcolor:'rgba(0,0,0,0)'}},{responsive:true,displaylogo:false});
    }
    function drawGammaTable(){const rows=D.arms[arm.value].rounds[round.value].gamma;q('gs-gamma-table').innerHTML=D.gammas.map(g=>{const r=rows[g];return`<tr><td>${g==='1'?'1.0':g}</td><td>${fmt(r.sr)}</td><td>${fmt(r.cr)}</td><td>${fmt(r.oob)}</td><td>${fmt(r.validity)}</td><td>${fmt(r.clearance)}</td><td>${fmt(r.ttg,2)}</td><td>${r.routes.join('/')}</td></tr>`}).join('')}
    function draw(){draw3d();drawMinis();drawGammaTable()}arm.addEventListener('change',()=>{fillRounds();draw()});[round,gamma,outcome,mode,detour,overlay].forEach(element=>element.addEventListener('change',draw));fillRounds();draw();
  })();
  </script>
</div>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", type=_arm, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--old-bounds", type=_parse_bounds, default=_parse_bounds("-2.5,1.3,-1.7,1.8,0.1,1.7"))
    parser.add_argument("--rounds", type=int, nargs="+")
    parser.add_argument("--reference", type=_reference, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task = json.loads(args.task_config.read_text())
    geometry = _geometry(task)
    new_bounds = np.asarray(geometry["bounds"], np.float64)
    requested_rounds = None if args.rounds is None else set(args.rounds)
    arms, inputs = {}, [args.task_config]
    for label, eval_dir in args.arm:
        arms[label], arm_inputs = _load_arm(
            eval_dir, args.old_bounds, new_bounds, requested_rounds,
        )
        inputs.extend(arm_inputs)
    reference_label, reference_round = args.reference
    if reference_label not in arms or str(reference_round) not in arms[reference_label]["rounds"]:
        raise ValueError("reference arm/round is not present in the selected data")
    payload = {
        **geometry,
        "oldBounds": args.old_bounds.tolist(),
        "newBounds": new_bounds.tolist(),
        "gammas": [f"{float(value):g}" for value in task["data"]["gammas"]],
        "reference": {"arm": reference_label, "round": reference_round},
        "arms": arms,
    }
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    fragment = FRAGMENT.replace("__DATA__", data)
    if re.search(r"<(?:!doctype|html|head|body)\b", fragment, flags=re.I):
        raise ValueError("visualization must remain an HTML fragment")
    if len(fragment.encode()) >= 1024 * 1024:
        raise ValueError("interactive fragment exceeds the 1 MiB budget")
    _atomic_write(args.output, fragment)
    provenance_path = args.output.with_suffix(".provenance.json")
    provenance = {
        "kind": "goal-space old/new boundary detour comparison",
        "generated_unix": time.time(),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "reference": payload["reference"],
        "round_filter": sorted(requested_rounds) if requested_rounds else None,
        "inputs_sha256": {
            str(path.resolve()): _sha256(path) for path in dict.fromkeys(inputs)
        },
        "path_compaction": (
            f"dense rollout paths downsampled to {MAX_POINTS} points plus "
            "old-y crossing, turnaround, and new-exit neighbors"
        ),
    }
    _atomic_write(provenance_path, json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "bytes": args.output.stat().st_size,
        "provenance": str(provenance_path.resolve()),
        "arms": list(arms),
    }, indent=2))


if __name__ == "__main__":
    main()
