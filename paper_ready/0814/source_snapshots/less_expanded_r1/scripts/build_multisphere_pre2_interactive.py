#!/usr/bin/env python3
"""Build the PRE2 five-scene H_P / K-B / gamma interactive fragment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def _round(values, decimals=3):
    return np.round(np.asarray(values), decimals).tolist()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _even(rows, count=6):
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count).round().astype(int)
    return [rows[int(index)] for index in np.unique(indices)]


def _path(values, max_points=100):
    points = np.asarray(values, np.float32).reshape(-1, 3)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).round().astype(int)
        points = points[np.unique(indices)]
    return _round(points, 3)


def _compact_summary(summary):
    keys = (
        "episodes", "status_counts", "SR", "CR", "OOB", "NVP", "timeout",
        "mean_min_clearance_m",
        "all_k_valid_rate", "selected_b_valid_rate",
        "selected_b_recall_of_all_k_positives",
        "all_k_progress_only_reject_rate", "all_k_tail_oob_rate",
        "selected_b_tail_oob_rate", "chosen_tail_oob_rate",
        "initial_sigma_tied_rate", "mean_command_rms",
        "mean_command_peak", "mean_command_saturation_rate",
        "mean_command_jerk",
    )
    return {key: summary.get(key) for key in keys}


def _trajectory_separation(scene_rows) -> dict:
    successful = [row for row in scene_rows if row["status"] == "SUCCESS"]
    signatures = {}
    start = np.asarray([-2.1, 1.5, 0.9], np.float64)
    goal = np.asarray([0.7, -1.5, 0.9], np.float64)
    forward = goal - start
    forward /= np.linalg.norm(forward)
    lateral = np.asarray([-forward[1], forward[0], 0.0])
    for row in scene_rows:
        path = np.asarray(row["path"], np.float64).reshape(-1, 3)
        displacement = path - start
        longitudinal = displacement @ forward
        total = float((goal - start) @ forward)
        mask = (longitudinal >= 0.2 * total) & (longitudinal <= 0.8 * total)
        middle = displacement[mask] if bool(mask.any()) else displacement
        signatures[f"{float(row['gamma']):g}"] = {
            "status": row["status"],
            "mean_lateral_m": float(np.mean(middle @ lateral)),
            "mean_vertical_m": float(np.mean(middle[:, 2])),
        }
    pairwise = []
    for first in range(len(successful)):
        for second in range(first + 1, len(successful)):
            first_path = np.asarray(successful[first]["path"], np.float64)
            second_path = np.asarray(successful[second]["path"], np.float64)
            grid = np.linspace(0.0, 1.0, 64)
            first_time = np.linspace(0.0, 1.0, len(first_path))
            second_time = np.linspace(0.0, 1.0, len(second_path))
            aligned_first = np.column_stack([
                np.interp(grid, first_time, first_path[:, axis])
                for axis in range(3)
            ])
            aligned_second = np.column_stack([
                np.interp(grid, second_time, second_path[:, axis])
                for axis in range(3)
            ])
            pairwise.append(float(np.mean(np.linalg.norm(
                aligned_first - aligned_second, axis=1,
            ))))
    return {
        "successful_gamma_count": len(successful),
        "signatures": signatures,
        "mean_pairwise_path_separation_m": (
            float(np.mean(pairwise)) if pairwise else None
        ),
        "minimum_pairwise_path_separation_m": (
            float(np.min(pairwise)) if pairwise else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = torch.load(
        args.preflight, map_location="cpu", weights_only=False,
    )
    diagnostic = "fixed_gallery" in artifact
    gallery = (
        artifact["fixed_gallery"]
        if diagnostic else artifact["matched_gallery"]
    )
    quota = artifact.get("true_dr_quota")
    rows = gallery["rows"]
    scenes = []
    for source in gallery["scenes"]:
        scene_index = int(source["scene_index"])
        scene_rows = [
            row for row in rows
            if row["scene"]["scene_hash"] == source["scene_hash"]
        ]
        trajectories = {
            f"{float(row['gamma']):g}": {
                "status": row["status"],
                "steps": int(row["steps"]),
                "path": _path(row["path"]),
                "diagnostics": {
                    key: row["diagnostics"].get(key)
                    for key in (
                        "all_k_valid_rate", "selected_b_valid_rate",
                        "all_k_tail_oob_rate", "chosen_tail_oob_rate",
                    )
                },
            }
            for row in scene_rows
        }
        gamma_half = next(
            row for row in scene_rows if np.isclose(row["gamma"], 0.5)
        )
        frames = []
        frame_count = 4 if diagnostic else 6
        for source_frame in _even(gamma_half["frames"], frame_count):
            hp = source_frame["hp"]
            hp_points = np.asarray(hp["points"])[::4]
            hp_values = np.asarray(hp["clipped_hp"])[::4]
            hp_inside = np.asarray(hp["inside_taskspace"])[::4]
            hp_obstacle = np.asarray(hp["inside_obstacle"])[::4]
            geometry = source_frame["candidate_geometry"]
            frames.append({
                "step": int(source_frame["control_step"]),
                "robot": _round(source_frame["robot"][:3], 3),
                "paths": [_path(value["path"], 20) for value in geometry],
                "tail_oob": [bool(value["tail_oob"]) for value in geometry],
                "costs": [value["cost_breakdown"] for value in geometry],
                "sigma_k": _round(source_frame["sigma_K"], 6),
                "selected": np.asarray(source_frame["selected"], int).tolist(),
                "selected_sigma": _round(
                    source_frame["selected_sigma"], 6,
                ),
                "verification": source_frame["verification_all_K"],
                "chosen": source_frame["chosen"],
                "hp": {
                    "points": _round(hp_points, 3),
                    "values": _round(hp_values, 4),
                    "inside_taskspace": hp_inside.astype(int).tolist(),
                    "inside_obstacle": hp_obstacle.astype(int).tolist(),
                    "full_grid_shape": list(hp["full_grid_shape"]),
                    "max_error": float(hp["full_grid_max_abs_error"]),
                    "active_faces": int(hp["active_dynamic_faces"]),
                    "query_center_collision": bool(
                        hp["query_center_collision"]
                    ),
                },
            })
        scenes.append({
            "index": scene_index,
            "hash": source["scene_hash"],
            "law": source.get("law"),
            "anchor": source.get("geometry", {}).get("anchor_center_m"),
            "progress_separation": source.get("diversity", {}).get(
                "mean_pairwise_success_progress_lateral_z_distance_m"
            ),
            "signatures": source.get("diversity", {}).get(
                "signatures", {}
            ),
            "spheres": _round(source["spheres"], 4),
            "trajectories": trajectories,
            "separation": _trajectory_separation(scene_rows),
            "frames": frames,
        })
    if diagnostic:
        quota_payload = {
            "diagnostic": True,
            "complete": False,
            "summaries": {
                key: _compact_summary(value)
                for key, value in gallery["by_gamma"].items()
            },
        }
    else:
        quota_payload = {
            "diagnostic": False,
            "complete": bool(quota["complete"]),
            "quota": int(quota["quota_per_gamma"]),
            "retries": quota["retry_batches_used"],
            "summaries": {
                key: _compact_summary(value)
                for key, value in quota["summary"].items()
            },
            "scene_hashes_pairwise_disjoint": bool(
                quota["scene_hashes_pairwise_disjoint"]
            ),
        }
    payload = {
        "contract": artifact["contract"],
        "arm": artifact.get("arm", artifact["contract"].get("arm")),
        "diagnostic": diagnostic,
        "title": (
            "PRE2 · dense-z paired-γ diagnostic"
            if diagnostic else "PRE2 · six-sphere Round-1 preflight"
        ),
        "subtitle": (
            "Shared scene and paired K bases; sphere centers use "
            "z ~ Uniform[0.7, 1.1]."
            if diagnostic else
            "True-DR quota evidence and fixed-scene γ counterfactuals "
            "are kept separate."
        ),
        "taskspace": {
            "bounds": [[-2.5, 1.3], [-1.7, 1.8], [0.1, 1.7]],
            "start": [-2.1, 1.5, 0.9],
            "goal": [0.7, -1.5, 0.9],
            "physical_radius": 0.1905,
            "effective_radius": 0.2405,
        },
        "quota": quota_payload,
        "gallery_summary": _compact_summary(gallery["summary"]),
        "gallery_gamma": {
            key: _compact_summary(value)
            for key, value in gallery["by_gamma"].items()
        },
        "scenes": scenes,
    }
    data = json.dumps(
        _json_ready(payload), separators=(",", ":"), allow_nan=False,
    )
    fragment = r'''<meta charset="utf-8">
<div id="pre2-multisphere-lab" class="viz-shell">
  <style>
    #pre2-multisphere-lab {
      --bg:var(--color-background-primary,#fff);
      --surface:var(--color-background-secondary,#f5f7fa);
      --text:var(--color-text-primary,#182033);
      --muted:var(--color-text-secondary,#5d6677);
      --border:var(--color-border-secondary,#d8dde6);
      --viz-blue:var(--blue,#2877d4);
      --viz-green:var(--green,#208967);
      --viz-orange:var(--orange,#d66a3c);
      --viz-red:var(--red,#bf3c42);
      --viz-purple:var(--purple,#8b5bc1);
      --viz-yellow:var(--yellow,#ad7800);
      width:100%; min-height:680px; color:var(--text); background:transparent;
      font:13px/1.42 ui-sans-serif,system-ui,sans-serif;
    }
    #pre2-multisphere-lab *{box-sizing:border-box}
    #pre2-multisphere-lab h2{font-size:19px;margin:0 0 4px}
    #pre2-multisphere-lab p{margin:0;color:var(--muted)}
    #pre2-multisphere-lab .top{display:flex;justify-content:space-between;gap:14px;align-items:flex-start;margin-bottom:9px;padding:8px 10px;border:1px solid var(--border);border-radius:8px;background:var(--surface)}
    #pre2-multisphere-lab .pill{border:1px solid var(--border);border-radius:999px;padding:5px 9px;background:var(--surface);white-space:nowrap}
    #pre2-multisphere-lab .pill.ok{border-color:var(--viz-green);color:var(--viz-green)}
    #pre2-multisphere-lab .pill.bad{border-color:var(--viz-red);color:var(--viz-red)}
    #pre2-multisphere-lab .quota-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;margin:0 0 8px}
    #pre2-multisphere-lab .qcard{border:1px solid var(--border);border-radius:8px;background:var(--surface);padding:7px 9px}
    #pre2-multisphere-lab .qcard b{font-size:15px;display:block;font-weight:600}
    #pre2-multisphere-lab .toolbar{display:grid;grid-template-columns:auto 1fr;gap:8px;align-items:center;margin:0 0 8px}
    #pre2-multisphere-lab .scene-tabs,.layers{display:flex;gap:6px;flex-wrap:wrap}
    #pre2-multisphere-lab button{border:1px solid var(--border);border-radius:7px;background:var(--surface);color:var(--text);padding:6px 9px;cursor:pointer}
    #pre2-multisphere-lab button[aria-pressed="true"]{border-color:var(--viz-blue);color:var(--viz-blue)}
    #pre2-multisphere-lab .stage{border:1px solid var(--border);border-radius:10px;overflow:hidden;background:var(--bg)}
    #pre2-multisphere-lab #plot{height:510px;width:100%}
    #pre2-multisphere-lab .slider{display:grid;grid-template-columns:auto 1fr auto auto;gap:9px;align-items:center;padding:8px 10px;border-top:1px solid var(--border);background:var(--surface)}
    #pre2-multisphere-lab input{width:100%}
    #pre2-multisphere-lab .readout{font-variant-numeric:tabular-nums;min-width:95px;text-align:right}
    #pre2-multisphere-lab .detail{display:grid;grid-template-columns:1.2fr 1fr;gap:8px;margin-top:8px}
    #pre2-multisphere-lab .panel{border:1px solid var(--border);background:var(--surface);border-radius:8px;padding:8px 10px;min-height:78px}
    #pre2-multisphere-lab .panel strong{font-weight:600}
    #pre2-multisphere-lab .legend{display:flex;gap:10px;flex-wrap:wrap;margin-top:6px;color:var(--muted);font-size:12px}
    #pre2-multisphere-lab .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
    #pre2-multisphere-lab .foot{font-size:12px;margin-top:7px}
    @media(prefers-color-scheme:dark){
      #pre2-multisphere-lab{
        --bg:var(--color-background-primary,#171717);
        --surface:var(--color-background-secondary,#252525);
        --text:var(--color-text-primary,#f4f4f4);
        --muted:var(--color-text-secondary,#b7bdc8);
        --border:var(--color-border-secondary,#4b4b4b);
        --viz-blue:var(--blue,#6da9ff);
        --viz-green:var(--green,#55c99c);
        --viz-orange:var(--orange,#ff9b75);
        --viz-red:var(--red,#ff737b);
        --viz-purple:var(--purple,#c18df0);
        --viz-yellow:var(--yellow,#f3bd3e);
      }
    }
    @media(max-width:650px){
      #pre2-multisphere-lab .quota-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
      #pre2-multisphere-lab .toolbar{grid-template-columns:1fr}
      #pre2-multisphere-lab #plot{height:420px}
      #pre2-multisphere-lab .detail{grid-template-columns:1fr}
      #pre2-multisphere-lab .top{flex-direction:column}
    }
  </style>
  <div class="top"><div><h2 id="title"></h2><p id="subtitle"></p></div><span id="gate" class="pill"></span></div>
  <div id="quota-grid" class="quota-grid"></div>
  <div class="toolbar"><div id="scene-tabs" class="scene-tabs"></div><div class="layers"><button data-layer="trajectories" aria-pressed="true">four γ paths</button><button data-layer="branches" aria-pressed="true">K/B branches</button><button data-layer="hp" aria-pressed="true">exact H<sub>P</sub></button><button data-layer="physical" aria-pressed="true">physical shells</button></div></div>
  <div class="stage"><div id="plot" role="img" aria-label="Interactive 3D multi-sphere policy, H P grid and uncertainty acquired candidate paths"></div><div class="slider"><button id="prev" aria-label="Previous gamma 0.5 control step">◀</button><input id="frame" type="range" min="0" step="1" value="0" aria-label="Gamma 0.5 control step"><button id="next" aria-label="Next gamma 0.5 control step">▶</button><span id="frame-label" class="readout"></span></div></div>
  <div class="detail"><div class="panel" id="candidate-detail"></div><div class="panel" id="hp-detail"></div></div>
  <div class="legend"><span><i class="dot" style="background:var(--viz-yellow)"></i>chosen</span><span><i class="dot" style="background:var(--viz-green)"></i>B execution-eligible</span><span><i class="dot" style="background:var(--viz-orange)"></i>B progress/target-ineligible</span><span><i class="dot" style="background:var(--viz-red)"></i>B verifier-invalid</span><span><i class="dot" style="background:var(--viz-blue)"></i>unqueried K offline verifier-valid</span><span><i class="dot" style="background:var(--muted)"></i>unqueried K offline verifier-invalid</span><span>H<sub>P</sub>: red &lt; 0, purple 0–0.75, blue &gt; 0.75; opacity rises as H<sub>P</sub> falls</span></div>
  <p class="foot" id="foot"></p>
</div>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script>
(()=>{
 const D=__DATA__,root=document.getElementById('pre2-multisphere-lab'),css=getComputedStyle(root),C=n=>css.getPropertyValue(n).trim();
 let sceneIndex=0,frameIndex=0,layers={trajectories:true,branches:true,hp:true,physical:true};
 const tabs=root.querySelector('#scene-tabs'),slider=root.querySelector('#frame'),plot=root.querySelector('#plot');
 const gammaColors={'0.1':C('--viz-blue'),'0.3':C('--viz-green'),'0.5':C('--viz-purple'),'1':C('--viz-orange')};
 root.querySelector('#title').textContent=D.title;root.querySelector('#subtitle').textContent=D.subtitle;
 const gate=root.querySelector('#gate');gate.textContent=D.diagnostic?'DIAGNOSTIC · no update/no replay':(D.quota.complete?'GO · q2 reached':'NO-GO · q2 incomplete');gate.className='pill '+(D.diagnostic?'':(D.quota.complete?'ok':'bad'));
 const pct=v=>v==null?'—':(100*v).toFixed(1)+'%';
 Object.keys(D.quota.summaries).sort((a,b)=>Number(a)-Number(b)).forEach(g=>{const m=D.quota.summaries[g],q=document.createElement('div');q.className='qcard';if(D.diagnostic){const ok=m.status_counts?m.status_counts.SUCCESS:Math.round(m.SR*m.episodes);q.innerHTML=`γ ${g}<b>${ok}/${m.episodes} success · ${pct(m.SR)} SR</b><span>OOB ${pct(m.OOB)} · NVP ${pct(m.NVP)} · tail ${pct(m.chosen_tail_oob_rate)}</span>`}else{const batches=D.quota.retries[g];q.innerHTML=`γ ${g}<b>${pct(m.SR)} SR · retry ${Math.max(0,batches-1)}</b><span>${batches} batch · OOB ${pct(m.OOB)} · NVP ${pct(m.NVP)} · tail ${pct(m.chosen_tail_oob_rate)}</span>`}root.querySelector('#quota-grid').appendChild(q)});
 D.scenes.forEach((s,i)=>{const b=document.createElement('button');const label=D.diagnostic?s.index:i+1;b.textContent='Scene '+label+(s.anchor?' · anchor':'');b.setAttribute('aria-pressed',i===0?'true':'false');b.onclick=()=>{sceneIndex=i;frameIndex=0;[...tabs.children].forEach((x,j)=>x.setAttribute('aria-pressed',j===i?'true':'false'));render()};tabs.appendChild(b)});
 root.querySelectorAll('[data-layer]').forEach(b=>b.onclick=()=>{const k=b.dataset.layer;layers[k]=!layers[k];b.setAttribute('aria-pressed',layers[k]?'true':'false');render()});
 const xyz=p=>({x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2])});
 function joined(paths){const x=[],y=[],z=[];paths.forEach(p=>{p.forEach(v=>{x.push(v[0]);y.push(v[1]);z.push(v[2])});x.push(null);y.push(null);z.push(null)});return{x,y,z}}
 function sphere(c,r,color,opacity,name){const x=[],y=[],z=[];for(let i=0;i<=10;i++){const ph=Math.PI*i/10,X=[],Y=[],Z=[];for(let j=0;j<=16;j++){const th=2*Math.PI*j/16;X.push(c[0]+r*Math.sin(ph)*Math.cos(th));Y.push(c[1]+r*Math.sin(ph)*Math.sin(th));Z.push(c[2]+r*Math.cos(ph))}x.push(X);y.push(Y);z.push(Z)}return{type:'surface',x,y,z,showscale:false,opacity,colorscale:[[0,color],[1,color]],hoverinfo:'skip',name}}
 function bounds(){const b=D.taskspace.bounds,c=[];for(const x of b[0])for(const y of b[1])for(const z of b[2])c.push([x,y,z]);const e=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]],p=[];e.forEach(v=>{p.push(c[v[0]],c[v[1]],[null,null,null])});return{type:'scatter3d',mode:'lines',...xyz(p),line:{color:C('--muted'),width:2},opacity:.35,hoverinfo:'skip',name:'taskspace'}}
 function render(){const s=D.scenes[sceneIndex],frames=s.frames,f=frames[Math.min(frameIndex,frames.length-1)],tr=[bounds()];slider.max=Math.max(frames.length-1,0);slider.value=frameIndex;root.querySelector('#frame-label').textContent=frames.length?`step ${f.step} · ${frameIndex+1}/${frames.length}`:'no frames';
  s.spheres.forEach(v=>{tr.push(sphere(v.slice(0,3),D.taskspace.effective_radius,C('--viz-orange'),.13,'effective'));if(layers.physical)tr.push(sphere(v.slice(0,3),D.taskspace.physical_radius,C('--viz-orange'),.35,'physical'))});
  if(layers.trajectories)Object.entries(s.trajectories).sort((a,b)=>Number(a[0])-Number(b[0])).forEach(([g,t])=>{const success=t.status==='SUCCESS';tr.push({type:'scatter3d',mode:'lines',...xyz(t.path),line:{color:gammaColors[g],width:g==='0.5'?7:5,dash:success?'solid':'dash'},opacity:success?.95:.32,name:`γ ${g} · ${t.status}`,hovertemplate:`γ ${g} · ${t.status}<extra></extra>`})});
  if(f&&layers.hp){const p=f.hp.points,v=f.hp.values,colors=v.map(h=>{const a=.08+.72*(1-h)/2;if(h<0)return `rgba(218,63,72,${a})`;if(h>.75)return `rgba(55,128,220,${a})`;return `rgba(142,91,193,${a})`});tr.push({type:'scatter3d',mode:'markers',...xyz(p),marker:{size:2.4,color:colors},customdata:v.map((h,i)=>[h,f.hp.inside_taskspace[i],f.hp.inside_obstacle[i]]),hovertemplate:'H_P=%{customdata[0]:.3f}<br>inside box=%{customdata[1]}<br>inside obstacle=%{customdata[2]}<extra></extra>',name:'clipped H_P'})}
  if(f&&layers.branches){const groups={chosen:[],good:[],progress:[],bad:[],offlineGood:[],offlineBad:[]},selected=new Set(f.selected);f.paths.forEach((p,i)=>{const q=f.verification[i];if(i===f.chosen)groups.chosen.push(p);else if(selected.has(i)){if(!q.valid)groups.bad.push(p);else if(!q.progress_eligible||!q.target_eligible)groups.progress.push(p);else groups.good.push(p)}else if(q.valid)groups.offlineGood.push(p);else groups.offlineBad.push(p)});const specs={chosen:[C('--viz-yellow'),7,1],good:[C('--viz-green'),4,.85],progress:[C('--viz-orange'),4,.85],bad:[C('--viz-red'),4,.8],offlineGood:[C('--viz-blue'),2,.25],offlineBad:[C('--muted'),1.5,.18]};Object.entries(groups).forEach(([k,p])=>{if(!p.length)return;const a=joined(p),sp=specs[k];tr.push({type:'scatter3d',mode:'lines',...a,line:{color:sp[0],width:sp[1]},opacity:sp[2],hoverinfo:'skip',name:k})});tr.push({type:'scatter3d',mode:'markers',x:[f.robot[0]],y:[f.robot[1]],z:[f.robot[2]],marker:{size:5,color:C('--viz-yellow')},name:'query robot'})}
  if(s.anchor)tr.push({type:'scatter3d',mode:'markers',x:[s.anchor[0]],y:[s.anchor[1]],z:[s.anchor[2]],marker:{size:8,color:C('--viz-yellow'),symbol:'diamond',line:{color:C('--text'),width:2}},name:'s=.2 anchor',hovertemplate:'fixed s=.20 anchor<extra></extra>'});
  tr.push({type:'scatter3d',mode:'markers',x:[D.taskspace.start[0],D.taskspace.goal[0]],y:[D.taskspace.start[1],D.taskspace.goal[1]],z:[D.taskspace.start[2],D.taskspace.goal[2]],marker:{size:[5,7],color:[C('--text'),C('--viz-yellow')],symbol:['square','diamond']},text:['start','goal'],hovertemplate:'%{text}<extra></extra>',name:'endpoints'});
  Plotly.react(plot,tr,{margin:{l:0,r:0,t:6,b:0},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:C('--text')},showlegend:false,scene:{xaxis:{title:'x',range:D.taskspace.bounds[0],gridcolor:C('--border')},yaxis:{title:'y',range:D.taskspace.bounds[1],gridcolor:C('--border')},zaxis:{title:'z',range:D.taskspace.bounds[2],gridcolor:C('--border')},aspectmode:'manual',aspectratio:{x:1.08,y:1,z:.55},camera:{eye:{x:1.45,y:1.5,z:1.05}}}},{responsive:true,displaylogo:false});
  if(f){const ch=f.chosen,c=ch==null?null:f.costs[ch],vs=f.selected_sigma.map((v,i)=>`${i+1}:${v.toFixed(3)}`).join(' · '),sep=s.separation.mean_pairwise_path_separation_m,statuses=Object.entries(s.trajectories).sort((a,b)=>Number(a[0])-Number(b[0])).map(([g,t])=>`γ${g}:${t.status}`).join(' · '),progressSep=s.progress_separation,zsig=Object.entries(s.signatures||{}).sort((a,b)=>Number(a[0])-Number(b[0])).filter(([,v])=>v.mean_z_m!=null).map(([g,v])=>`γ${g}:${v.mean_z_m.toFixed(3)}`).join(' · '),diversity=D.diagnostic?`progress-aligned lateral–z separation ${progressSep==null?'—':progressSep.toFixed(3)+' m'} · mean z ${zsig||'—'}`:`time-aligned success-path separation ${sep==null?'—':sep.toFixed(3)+' m'}`;root.querySelector('#candidate-detail').innerHTML=`<strong>γ=.5 acquisition · buffer 0</strong><br>K=16 → B=8 · selected σ order ${vs}<br>${ch==null?'No execution-eligible B candidate':`chosen K[${ch}] · cost ${c.total.toFixed(2)} = native ${c.base_native.toFixed(2)} + wall ${c.interior_wall.toFixed(2)} + axis ${c.axis_cylinder.toFixed(2)} + Δcontrol ${c.control_override_delta.toFixed(2)} · tail OOB ${f.tail_oob[ch]}`}<br>${statuses}<br>successful γ ${s.separation.successful_gamma_count}/4 · ${diversity}`;root.querySelector('#hp-detail').innerHTML=`<strong>Exact encoder H<sub>P</sub></strong><br>${f.hp.full_grid_shape.join('×')} · displayed 1/${4*4*5*4} of cells<br>raster/direct max error ${f.hp.max_error.toExponential(2)} · active dynamic faces ${f.hp.active_faces}<br>query center collision ${f.hp.query_center_collision}`}
 }
 root.querySelector('#foot').innerHTML=(D.diagnostic?'Paired K bases isolate γ-conditioning within each fixed scene; paths are diagnostic and never enter replay. ':'The K labels outside selected B come from an offline all-K diagnostic oracle; execution queried only B=8. ')+'At buffer 0, initial σ<sub>K</sub> is tied: draw 1 is uniform and later draws provide conditional within-K diversity, not PRE-relative epistemic novelty. H<sub>P</sub> is the exact 1×32×32×100 encoder field (display-decimated). Candidate color shows the aggregate verifier result; chosen H-tail taskspace OOB is reported separately.';
 slider.oninput=()=>{frameIndex=+slider.value;render()};root.querySelector('#prev').onclick=()=>{frameIndex=Math.max(0,frameIndex-1);render()};root.querySelector('#next').onclick=()=>{const n=D.scenes[sceneIndex].frames.length;frameIndex=Math.min(Math.max(0,n-1),frameIndex+1);render()};render();
})();
</script>'''.replace('__DATA__', data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment, encoding="utf-8")
    size = args.output.stat().st_size
    if size >= 1_000_000:
        raise RuntimeError(f"visual fragment is too large: {size} bytes")
    print(f"{args.output} ({size} bytes)")


if __name__ == "__main__":
    main()
