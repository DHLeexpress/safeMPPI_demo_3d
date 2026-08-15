#!/usr/bin/env python3
"""Render H10 goal-box fixed-bank trajectories and first OOB exits in 3-D."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch


DEFAULT_STAGE = Path(
    "/Users/dhl/Documents/safeMPPI_demo_3d/results/"
    "stage1_single_ball_t128/0811_pre2_h10_goalbox_sweep"
)
MAX_PATH_POINTS = 14
FACES = ("x-min", "x-max", "y-min", "y-max", "z-min", "z-max")


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


def _compact_row(row: dict, bounds: np.ndarray) -> dict:
    states = np.asarray(row["states"], dtype=np.float64)
    dense_steps = np.asarray(row.get("dense_steps", ()), dtype=np.float64)
    dense_path = (
        np.concatenate([states[:1, :3], dense_steps.reshape(-1, 3)])
        if dense_steps.size
        else states[:, :3]
    )
    exit_index, exit_face = _first_exit(dense_path, bounds)
    indices = set(np.linspace(
        0, len(dense_path) - 1,
        min(MAX_PATH_POINTS, len(dense_path)), dtype=int,
    ).tolist())
    if exit_index is not None:
        indices.add(exit_index)
        indices.add(max(0, exit_index - 1))
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
    return {
        "n": int(row["episodes"]),
        "sr": round(float(row["SR"]), 6),
        "cr": round(float(row["CR"]), 6),
        "oob": round(float(row["OOB"]), 6),
        "to": round(float(row["timeout"]), 6),
        "v": round(float(row["window_validity"]), 6),
        "cov": round(float(row.get("route_coverage", 0.0)), 6),
        "routes": [
            int(route_counts.get(name, 0))
            for name in ("below", "above", "left", "right")
        ],
        "clr": (
            None
            if row.get("successful_min_clearance_m") is None
            else round(float(row["successful_min_clearance_m"]), 5)
        ),
        "ttg": (
            None
            if row.get("successful_time_to_goal_s") is None
            else round(float(row["successful_time_to_goal_s"]), 4)
        ),
    }


def _load_evaluation(raw_eval: Path, bounds: np.ndarray) -> dict:
    raw_trajectories = raw_eval.with_name("raw_trajectories.pt")
    if not raw_trajectories.is_file():
        raise FileNotFoundError(
            f"raw evaluation is not fully published: {raw_trajectories}"
        )
    summary = json.loads(raw_eval.read_text())["summary"]
    trajectories = torch.load(
        raw_trajectories, map_location="cpu", weights_only=False,
    )
    rounds = {}
    for round_key in sorted(summary, key=int):
        round_i = int(round_key)
        source_rows = trajectories.get(round_i, trajectories.get(round_key))
        if source_rows is None:
            raise KeyError(
                f"round {round_i} missing from {raw_trajectories}"
            )
        round_summary = summary[round_key]
        rounds[str(round_i)] = {
            "rows": [_compact_row(row, bounds) for row in source_rows],
            "pooled": _metric(round_summary["pooled"]),
            "gamma": {
                str(gamma): _metric(values)
                for gamma, values in round_summary["per_gamma"].items()
            },
        }
    return {
        "rounds": rounds,
        "raw_eval": str(raw_eval),
        "raw_trajectories": str(raw_trajectories),
    }


def _latest_complete_evaluation(output: Path) -> Path | None:
    candidates = []
    for raw_eval in output.glob("fixed_eval_r000_r*/raw_eval.json"):
        if not raw_eval.with_name("raw_trajectories.pt").is_file():
            continue
        summary = json.loads(raw_eval.read_text()).get("summary", {})
        if summary:
            candidates.append((max(map(int, summary)), raw_eval))
    return max(candidates, default=(None, None))[1]


def _task_geometry(task_config: dict) -> dict:
    task = task_config["taskspace"]
    bounds = [
        [float(origin), float(origin + size)]
        for origin, size in zip(task["origin"], task["size"])
    ]
    return {
        "bounds": bounds,
        "start": [float(value) for value in task["start"][:3]],
        "goal": [float(value) for value in task["goal"]],
        "sphere": [
            float(value) for value in task_config["obstacles"]["spheres"][0]
        ],
    }


def _build_payload(stage: Path) -> tuple[dict, list[Path], list[str]]:
    sweep_path = stage / "SWEEP.json"
    sweep = json.loads(sweep_path.read_text())
    task_path = Path(sweep["fixed"]["task_config"])
    geometry = _task_geometry(json.loads(task_path.read_text()))
    bounds = np.asarray(geometry["bounds"], dtype=np.float64)
    inputs = [sweep_path, task_path]
    skipped = []

    legacy_raw_eval = Path(sweep["fixed"]["legacy_brake100_r5"])
    legacy = _load_evaluation(legacy_raw_eval, bounds)
    inputs.extend([
        legacy_raw_eval, legacy_raw_eval.with_name("raw_trajectories.pt"),
    ])
    datasets = {
        "legacy_brake100": {
            "label": "legacy brake100",
            "kind": "legacy",
            **legacy,
        },
    }
    for arm in sweep["arms"]:
        raw_eval = _latest_complete_evaluation(Path(arm["output"]))
        if raw_eval is None:
            skipped.append(arm["name"])
            continue
        datasets[arm["name"]] = {
            "label": f"goal-box w={float(arm['weight']):g}",
            "kind": "goalbox",
            "weight": float(arm["weight"]),
            **_load_evaluation(raw_eval, bounds),
        }
        inputs.extend([raw_eval, raw_eval.with_name("raw_trajectories.pt")])

    goal_box = sweep["fixed"]["goal_centered_box"]
    payload = {
        **geometry,
        "goalBox": [goal_box["lower"], goal_box["upper"]],
        "reference": "legacy_brake100",
        "referenceRound": 5,
        "gammas": ["0.1", "0.3", "0.5", "1"],
        "datasets": datasets,
    }
    return payload, inputs, skipped


FRAGMENT = r'''
<div id="h10-goalbox-raw-lab">
  <style>
    #h10-goalbox-raw-lab { color: var(--foreground); font: 13px/1.35 ui-sans-serif, system-ui, sans-serif; width: 100%; min-width: 0; overflow: hidden; }
    #h10-goalbox-raw-lab * { box-sizing: border-box; }
    #h10-goalbox-raw-lab .title { font-size: 20px; font-weight: 680; letter-spacing: -.02em; margin-bottom: 3px; }
    #h10-goalbox-raw-lab .sub { color: var(--muted-foreground); margin-bottom: 9px; }
    #h10-goalbox-raw-lab .controls { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: end; padding: 8px 0 10px; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
    #h10-goalbox-raw-lab .control { display: grid; gap: 3px; min-width: 105px; }
    #h10-goalbox-raw-lab label { color: var(--muted-foreground); font-size: 11px; font-weight: 650; letter-spacing: .035em; text-transform: uppercase; }
    #h10-goalbox-raw-lab select { color: var(--foreground); background: transparent; border: 1px solid var(--border); border-radius: 5px; min-height: 30px; padding: 4px 7px; }
    #h10-goalbox-raw-lab .check { display: flex; align-items: center; gap: 6px; min-height: 30px; color: var(--foreground); font-size: 12px; font-weight: 500; letter-spacing: 0; text-transform: none; }
    #h10-goalbox-raw-lab .summary { display: flex; flex-wrap: wrap; gap: 6px 17px; padding: 9px 0 2px; min-height: 42px; }
    #h10-goalbox-raw-lab .summary span { white-space: nowrap; }
    #h10-goalbox-raw-lab .summary small { color: var(--muted-foreground); padding-left: 3px; }
    #h10-goalbox-raw-lab .plot3d { width: 100%; height: 555px; }
    #h10-goalbox-raw-lab .mini-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; border-top: 1px solid var(--border); padding-top: 10px; }
    #h10-goalbox-raw-lab .mini { min-width: 0; height: 280px; }
    @media (max-width: 620px) {
      #h10-goalbox-raw-lab .controls { display: grid; grid-template-columns: 1fr 1fr; }
      #h10-goalbox-raw-lab .control { min-width: 0; }
      #h10-goalbox-raw-lab select { width: 100%; }
      #h10-goalbox-raw-lab .plot3d { height: 420px; }
      #h10-goalbox-raw-lab .mini-grid { grid-template-columns: 1fr; }
      #h10-goalbox-raw-lab .mini { height: 260px; }
    }
  </style>
  <div class="title">H10 goal-box: raw trajectories and first OOB exits</div>
  <div class="sub">Fixed seed 91000 · NFE 12 · 40 episodes/γ · dotted overlay is legacy brake100 r5</div>
  <div class="controls">
    <div class="control"><label for="h10-dataset">Arm</label><select id="h10-dataset"></select></div>
    <div class="control"><label for="h10-round">Round</label><select id="h10-round"></select></div>
    <div class="control"><label for="h10-gamma">Gamma</label><select id="h10-gamma"><option value="all">all</option></select></div>
    <div class="control"><label for="h10-outcome">Outcome</label><select id="h10-outcome"><option value="all">all</option><option>SUCCESS</option><option>COLLISION</option><option>OOB</option><option>TIMEOUT</option></select></div>
    <div class="control"><label for="h10-route">Route</label><select id="h10-route"><option value="all">all</option><option>below</option><option>above</option><option>left</option><option>right</option><option>none</option></select></div>
    <div class="control"><label for="h10-wall">First OOB wall</label><select id="h10-wall"><option value="all">all</option><option>x-min</option><option>x-max</option><option>y-min</option><option>y-max</option><option>z-min</option><option>z-max</option></select></div>
    <label class="check"><input id="h10-overlay" type="checkbox" checked> overlay legacy r5</label>
  </div>
  <div id="h10-summary" class="summary" aria-live="polite"></div>
  <div id="h10-plot" class="plot3d" role="img" aria-label="Interactive 3D fixed-bank trajectories with first out-of-bounds exits"></div>
  <div class="mini-grid">
    <div id="h10-trend" class="mini" role="img" aria-label="Roundwise fixed-bank success collision out-of-bounds and validity rates"></div>
    <div id="h10-walls" class="mini" role="img" aria-label="First out-of-bounds wall counts by round"></div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
  <script>
  (() => {
    const root=document.getElementById('h10-goalbox-raw-lab'), D=__DATA__;
    const css=getComputedStyle(root), token=n=>css.getPropertyValue(`--viz-series-${n}`).trim();
    const fg=css.getPropertyValue('--foreground').trim(), muted=css.getPropertyValue('--muted-foreground').trim(), border=css.getPropertyValue('--border').trim(), bg=css.getPropertyValue('--background').trim();
    const modeColor={below:token(1),above:token(2),left:token(3),right:token(4),none:muted};
    const statusColor={COLLISION:css.getPropertyValue('--destructive').trim(),OOB:token(5),TIMEOUT:muted};
    const wallColor={'x-min':token(1),'x-max':token(2),'y-min':token(5),'y-max':token(6),'z-min':token(3),'z-max':token(4)};
    const q=id=>root.querySelector(`#${id}`), dataset=q('h10-dataset'), round=q('h10-round'), gamma=q('h10-gamma'), outcome=q('h10-outcome'), route=q('h10-route'), wall=q('h10-wall'), overlay=q('h10-overlay');
    const fmt=(value,digits=3)=>value==null?'—':Number(value).toFixed(digits);
    const points=row=>{const out=[];for(let i=0;i<row.p.length;i+=3)out.push([row.p[i],row.p[i+1],row.p[i+2]]);return out};
    const keys=Object.keys(D.datasets), firstNew=keys.find(key=>D.datasets[key].kind!=='legacy');
    keys.forEach(key=>{const option=document.createElement('option');option.value=key;option.textContent=D.datasets[key].label;dataset.appendChild(option)});
    D.gammas.forEach(value=>{const option=document.createElement('option');option.value=value;option.textContent=value==='1'?'1.0':value;gamma.appendChild(option)});
    dataset.value=firstNew||D.reference;

    function fillRounds(){const prior=round.value, values=Object.keys(D.datasets[dataset.value].rounds).sort((a,b)=>+a-+b);round.replaceChildren();values.forEach(value=>{const option=document.createElement('option');option.value=value;option.textContent=`r${value}`;round.appendChild(option)});round.value=values.includes(prior)?prior:values.at(-1)}
    function metric(data,roundKey){const row=data.rounds[roundKey];return gamma.value==='all'?row.pooled:row.gamma[gamma.value]}
    function keep(row){return (gamma.value==='all'||String(row.g)===gamma.value)&&(outcome.value==='all'||row.s===outcome.value)&&(route.value==='all'||row.m===route.value)&&(wall.value==='all'||(row.x&&row.x.f===wall.value))}
    function boxTrace(bounds,color,dash='dot',width=3,name=null){const [xb,yb,zb]=bounds,vs=[[xb[0],yb[0],zb[0]],[xb[1],yb[0],zb[0]],[xb[0],yb[1],zb[0]],[xb[1],yb[1],zb[0]],[xb[0],yb[0],zb[1]],[xb[1],yb[0],zb[1]],[xb[0],yb[1],zb[1]],[xb[1],yb[1],zb[1]]],es=[[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]],x=[],y=[],z=[];es.forEach(([a,b])=>{[a,b].forEach(i=>{x.push(vs[i][0]);y.push(vs[i][1]);z.push(vs[i][2])});x.push(null);y.push(null);z.push(null)});return{type:'scatter3d',mode:'lines',x,y,z,line:{color,width,dash},hoverinfo:'skip',name:name||'',showlegend:Boolean(name)}}
    function sphereTrace(){const [cx,cy,cz,r]=D.sphere,x=[],y=[],z=[],i=[],j=[],k=[],nu=22,nv=13;for(let b=0;b<nv;b++){const p=Math.PI*b/(nv-1);for(let a=0;a<nu;a++){const t=2*Math.PI*a/nu;x.push(cx+r*Math.sin(p)*Math.cos(t));y.push(cy+r*Math.sin(p)*Math.sin(t));z.push(cz+r*Math.cos(p))}}for(let b=0;b<nv-1;b++)for(let a=0;a<nu;a++){const u=b*nu+a,v=b*nu+(a+1)%nu,w=(b+1)*nu+a,n=(b+1)*nu+(a+1)%nu;i.push(u,v);j.push(w,w);k.push(v,n)}return{type:'mesh3d',x,y,z,i,j,k,color:muted,opacity:.26,hoverinfo:'skip',name:'sphere',showlegend:true}}
    function joined(rows){const x=[],y=[],z=[],text=[];rows.forEach(row=>{const label=`γ=${row.g} · ${row.s} · ${row.m} · ep${row.e}${row.x?` · first exit ${row.x.f}`:''}`;points(row).forEach(p=>{x.push(p[0]);y.push(p[1]);z.push(p[2]);text.push(label)});x.push(null);y.push(null);z.push(null);text.push('')});return{x,y,z,text}}
    function pathTrace(rows,name,color,width=3,dash='solid',opacity=.65){return{type:'scatter3d',mode:'lines',...joined(rows),line:{color,width,dash},opacity,name,hovertemplate:'%{text}<extra></extra>',showlegend:true}}
    function draw3d(){const data=D.datasets[dataset.value], rows=data.rounds[round.value].rows.filter(keep), traces=[boxTrace(D.bounds,border),boxTrace([[D.goalBox[0][0],D.goalBox[1][0]],[D.goalBox[0][1],D.goalBox[1][1]],[D.goalBox[0][2],D.goalBox[1][2]]],token(6),'dash',2,'goal box'),sphereTrace()];
      if(overlay.checked&&dataset.value!==D.reference){const reference=D.datasets[D.reference].rounds[String(D.referenceRound)].rows.filter(keep);if(reference.length)traces.push(pathTrace(reference,'legacy brake100 r5',muted,2,'dot',.22))}
      const groups=new Map();rows.forEach(row=>{const key=row.s==='SUCCESS'?`SUCCESS · ${row.m}`:row.s;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(row)});groups.forEach((values,key)=>{const sample=values[0],color=sample.s==='SUCCESS'?modeColor[sample.m]:statusColor[sample.s];traces.push(pathTrace(values,key,color,sample.s==='OOB'?4:3,'solid',sample.s==='OOB'?.85:.58))});
      FACES.forEach(face=>{const exits=rows.filter(row=>row.x&&row.x.f===face);if(exits.length)traces.push({type:'scatter3d',mode:'markers',x:exits.map(row=>row.x.p[0]),y:exits.map(row=>row.x.p[1]),z:exits.map(row=>row.x.p[2]),text:exits.map(row=>`γ=${row.g} · ep${row.e} · ${face}`),hovertemplate:'first OOB %{text}<extra></extra>',marker:{size:5,color:wallColor[face],symbol:'x'},name:`exit ${face}`})});
      traces.push({type:'scatter3d',mode:'lines+markers',x:[D.start[0],D.goal[0]],y:[D.start[1],D.goal[1]],z:[D.start[2],D.goal[2]],line:{color:fg,width:2},marker:{size:[5,7],color:[fg,token(6)],symbol:['square','diamond']},text:['start','goal'],hovertemplate:'%{text}<extra></extra>',name:'start–goal'});
      Plotly.react(q('h10-plot'),traces,{margin:{l:0,r:0,t:8,b:0},paper_bgcolor:'rgba(0,0,0,0)',font:{color:fg,size:11},legend:{orientation:'h',y:1.02,x:0,bgcolor:'rgba(0,0,0,0)'},scene:{aspectmode:'data',xaxis:{title:'x [m]',range:D.bounds[0],backgroundcolor:bg,gridcolor:border},yaxis:{title:'y [m]',range:D.bounds[1],backgroundcolor:bg,gridcolor:border},zaxis:{title:'z [m]',range:D.bounds[2],backgroundcolor:bg,gridcolor:border},camera:{eye:{x:1.45,y:1.45,z:.9}}}},{responsive:true,displaylogo:false});
      const m=metric(data,round.value),ref=metric(D.datasets[D.reference],String(D.referenceRound)),walls={};rows.filter(row=>row.x).forEach(row=>walls[row.x.f]=(walls[row.x.f]||0)+1);const wallText=Object.entries(walls).map(([key,value])=>`${key} ${value}`).join(' · ')||'0';
      q('h10-summary').innerHTML=`<span>shown <strong>${rows.length}/${data.rounds[round.value].rows.length}</strong></span><span>SR <strong>${fmt(m.sr)}</strong><small>legacy ${fmt(ref.sr)}</small></span><span>CR <strong>${fmt(m.cr)}</strong><small>legacy ${fmt(ref.cr)}</small></span><span>OOB <strong>${fmt(m.oob)}</strong><small>legacy ${fmt(ref.oob)}</small></span><span>validity <strong>${fmt(m.v)}</strong><small>legacy ${fmt(ref.v)}</small></span><span>routes b/a/l/r <strong>${m.routes.join('/')}</strong></span><span>first exits <strong>${wallText}</strong></span><span>clearance <strong>${fmt(m.clr)} m</strong></span><span>TtG <strong>${fmt(m.ttg,2)} s</strong></span>`;
    }
    const FACES=['x-min','x-max','y-min','y-max','z-min','z-max'];
    function drawMinis(){const data=D.datasets[dataset.value],roundKeys=Object.keys(data.rounds).sort((a,b)=>+a-+b),xs=roundKeys.map(Number),rows=roundKeys.map(key=>metric(data,key)),series=[['SR','sr',token(3)],['CR','cr',token(2)],['OOB','oob',token(5)],['validity','v',token(4)]],traces=series.map(([name,key,color])=>({type:'scatter',mode:'lines+markers',name,x:xs,y:rows.map(row=>row[key]),line:{color,width:2},marker:{size:5},hovertemplate:`r%{x}<br>${name}: %{y:.3f}<extra></extra>`}));
      if(dataset.value!==D.reference){const ref=metric(D.datasets[D.reference],String(D.referenceRound));series.forEach(([name,key,color])=>traces.push({type:'scatter',mode:'markers',x:[D.referenceRound],y:[ref[key]],marker:{size:9,color,symbol:'diamond-open',line:{width:2,color}},showlegend:false,hovertemplate:`legacy r5 ${name}: %{y:.3f}<extra></extra>`}))}
      Plotly.react(q('h10-trend'),traces,{title:{text:`Roundwise fixed-bank rates · ${gamma.value==='all'?'pooled':`γ=${gamma.value}`}`,x:0,font:{size:14}},margin:{l:48,r:12,t:58,b:42},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:fg,size:11},xaxis:{title:'round',gridcolor:border,dtick:1},yaxis:{title:'rate',range:[0,1],gridcolor:border},legend:{orientation:'h',y:1.14,x:.2,bgcolor:'rgba(0,0,0,0)'}},{responsive:true,displaylogo:false});
      const wallTraces=FACES.map(face=>({type:'bar',name:face,x:xs,y:roundKeys.map(key=>data.rounds[key].rows.filter(row=>(gamma.value==='all'||String(row.g)===gamma.value)&&row.s==='OOB'&&row.x&&row.x.f===face).length),marker:{color:wallColor[face]},hovertemplate:`r%{x}<br>${face}: %{y}<extra></extra>`}));
      Plotly.react(q('h10-walls'),wallTraces,{barmode:'stack',title:{text:'OOB first-exit wall counts',x:0,font:{size:14}},margin:{l:48,r:12,t:58,b:42},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:fg,size:11},xaxis:{title:'round',gridcolor:border,dtick:1},yaxis:{title:'episodes',rangemode:'tozero',gridcolor:border},legend:{orientation:'h',y:1.16,x:0,bgcolor:'rgba(0,0,0,0)'}},{responsive:true,displaylogo:false});
    }
    function draw(){draw3d();drawMinis()}
    dataset.addEventListener('change',()=>{fillRounds();draw()});[round,gamma,outcome,route,wall,overlay].forEach(element=>element.addEventListener('change',draw));fillRounds();draw();
  })();
  </script>
</div>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        args.stage / "artifacts/h10_goalbox_raw_comparison.html"
    )
    payload, inputs, skipped = _build_payload(args.stage)
    data = json.dumps(
        payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    )
    fragment = FRAGMENT.replace("__DATA__", data)
    if re.search(r"<(?:!doctype|html|head|body)\b", fragment, flags=re.I):
        raise ValueError("visualization must remain an HTML fragment")
    _atomic_write(output, fragment)
    provenance_path = output.with_suffix(".provenance.json")
    provenance = {
        "kind": "H10 goal-box fixed-bank raw trajectory comparison",
        "generated_unix": time.time(),
        "stage": str(args.stage),
        "output": str(output),
        "output_sha256": _sha256(output),
        "included_datasets": list(payload["datasets"]),
        "skipped_incomplete_arms": skipped,
        "inputs": {
            str(path): _sha256(path) for path in dict.fromkeys(inputs)
        },
        "path_compaction": (
            f"actual dense raw-evaluation paths, evenly downsampled to "
            f"{MAX_PATH_POINTS} points plus exact first-exit neighbors"
        ),
    }
    _atomic_write(
        provenance_path,
        json.dumps(provenance, indent=2, allow_nan=False) + "\n",
    )
    print(
        f"wrote {output} ({output.stat().st_size} bytes); "
        f"datasets={','.join(payload['datasets'])}; "
        f"skipped={','.join(skipped) or 'none'}"
    )


if __name__ == "__main__":
    main()
