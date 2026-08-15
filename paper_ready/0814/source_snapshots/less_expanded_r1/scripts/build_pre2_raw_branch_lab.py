#!/usr/bin/env python3
"""Build the PRE2 raw-deployment 3-D failure laboratory fragment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _round(values, decimals=3):
    return np.round(np.asarray(values), decimals).tolist()


def _subset(indices, count):
    if len(indices) <= count:
        return indices
    return [indices[i] for i in np.linspace(0, len(indices) - 1, count).round().astype(int)]


def _case(row):
    frames = []
    for frame in row["frames"]:
        risks = frame["risks"]
        unsafe = [i for i, risk in enumerate(risks[1:], start=1) if risk["collision"] or risk["oob"]]
        safe = [i for i, risk in enumerate(risks[1:], start=1) if not risk["collision"] and not risk["oob"]]
        selected = [0, *_subset(unsafe, 9), *_subset(safe, 10)]
        selected = list(dict.fromkeys(selected))
        frames.append({
            "step": int(frame["step"]),
            "paths": [_round(frame["paths"][index][::3]) for index in selected],
            "risk": [risks[index] for index in selected],
            "raw_path_index": 0,
            "ensemble_safe": len(safe),
            "ensemble_unsafe": len(unsafe),
        })
    return {
        "episode": int(row["episode"]), "seed": int(row["seed"]),
        "status": row["status"], "mode": row["mode"], "steps": int(row["steps"]),
        "min_clearance_m": row["min_clearance_m"], "time_to_goal_s": row["time_to_goal_s"],
        "window_validity": float(row["window_validity"]),
        "path": _round(row["path"][::4]), "frames": frames,
        "spheres": _round(row["spheres"]), "cylinders": _round(row["cylinders"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = torch.load(args.input, map_location="cpu", weights_only=False)
    payload = {
        "contract": raw["contract"],
        "bounds": _round(raw["bounds"]), "start": _round(raw["start"]),
        "goal": _round(raw["goal"]), "summaries": raw["manifest_summaries"],
        "conditions": {
            "sphere_ood": {"label": "Sphere OOD", "cases": [_case(row) for row in raw["sphere_ood"]]},
            "cylinder_id": {"label": "Cylinder ID", "cases": [_case(row) for row in raw["cylinder_id"]]},
        },
    }
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    fragment = r'''<div id="pre2-raw-lab" class="viz-shell">
<style>
#pre2-raw-lab{--bg:var(--color-background-primary,light-dark(#fff,#181818));--surface:var(--color-background-secondary,light-dark(#f5f7fa,#252525));--text:var(--color-text-primary,light-dark(#172033,#fff));--muted:var(--color-text-secondary,light-dark(#5b6475,#b6bdc9));--border:var(--color-border-secondary,light-dark(#d6dbe5,#4b4b4b));--blue:var(--blue,light-dark(#2774d8,#6aa9ff));--green:var(--green,light-dark(#2a9d78,#55c99c));--red:var(--red,light-dark(#c23b3b,#ff7676));--gold:var(--yellow,light-dark(#a66a00,#f0b429));color:var(--text);font:13px/1.4 ui-sans-serif,system-ui,sans-serif;width:100%}#pre2-raw-lab *{box-sizing:border-box}#pre2-raw-lab h2{font-size:18px;margin:0 0 3px}#pre2-raw-lab p{margin:0;color:var(--muted)}#pre2-raw-lab .head{display:flex;justify-content:space-between;gap:14px;align-items:end;margin-bottom:10px}#pre2-raw-lab .tabs,#pre2-raw-lab .episodes{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}#pre2-raw-lab button{font:inherit;color:var(--text);background:var(--surface);border:1px solid var(--border);border-radius:7px;padding:6px 9px;cursor:pointer}#pre2-raw-lab button[aria-pressed=true]{border:2px solid var(--blue);padding:5px 8px}#pre2-raw-lab .episodes button.success{border-color:var(--green)}#pre2-raw-lab .episodes button.collision{border-color:var(--red)}#pre2-raw-lab .episodes button.oob{border-color:var(--gold)}#pre2-raw-lab .stage{border:1px solid var(--border);border-radius:10px;overflow:hidden;background:var(--bg)}#pre2-raw-lab #plot{width:100%;height:500px}#pre2-raw-lab .controls{display:grid;grid-template-columns:auto 1fr auto auto;gap:8px;align-items:center;padding:8px 10px;background:var(--surface);border-top:1px solid var(--border)}#pre2-raw-lab input{width:100%}#pre2-raw-lab .stats{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px;border-top:1px solid var(--border);padding:8px 10px;background:var(--surface)}#pre2-raw-lab .stat{color:var(--muted)}#pre2-raw-lab .stat b{display:block;color:var(--text);font-size:15px;font-weight:600}#pre2-raw-lab .legend{display:flex;gap:11px;flex-wrap:wrap;padding:7px 10px;border-top:1px solid var(--border);color:var(--muted)}#pre2-raw-lab .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}#pre2-raw-lab .why{margin-top:8px;font-size:12px}@media(max-width:620px){#pre2-raw-lab .head{display:block}#pre2-raw-lab #plot{height:400px}#pre2-raw-lab .stats{grid-template-columns:repeat(2,1fr)}#pre2-raw-lab .controls{grid-template-columns:auto 1fr auto}#pre2-raw-lab .frame{grid-column:1/-1}}
</style>
<div class="head"><div><h2>PRE2 raw-deployment failure laboratory</h2><p id="subtitle"></p></div><p id="audit"></p></div>
<div id="tabs" class="tabs"></div><div id="episodes" class="episodes" aria-label="Select raw rollout episode"></div>
<div class="stage"><div id="plot" role="img" aria-label="3-D raw policy H10 samples and executed rollout"></div><div class="controls"><button id="prev" aria-label="Previous branch snapshot">◀</button><input id="slider" type="range" min="0" step="1" value="0" aria-label="Branch snapshot"><button id="next" aria-label="Next branch snapshot">▶</button><span id="frame" class="frame"></span></div><div id="stats" class="stats"></div><div class="legend"><span><i class="dot" style="background:var(--gold)"></i>actual unfiltered K=1 H10 draw</span><span><i class="dot" style="background:var(--red)"></i>other H10 draws predicted collision/OOB</span><span><i class="dot" style="background:var(--blue)"></i>other predicted-safe H10 draws</span><span><i class="dot" style="background:var(--green)"></i>raw closed-loop path</span></div></div>
<p class="why"><b>Reading this:</b> the raw policy has no verifier or selection rule: each step samples one H10 plan at σ=1.0 and executes only its first action. Red is a diagnostic ensemble (64 extra draws), not a controller intervention. Switch scenes and branch times to separate lack of safe local support from an unsafe single draw.</p>
</div><script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script><script>(()=>{const D=__DATA__,root=document.getElementById('pre2-raw-lab'),C=n=>getComputedStyle(root).getPropertyValue(n).trim(),xyz=p=>({x:p.map(q=>q[0]),y:p.map(q=>q[1]),z:p.map(q=>q[2])}),join=ps=>{let x=[],y=[],z=[];ps.forEach(p=>{p.forEach(q=>{x.push(q[0]);y.push(q[1]);z.push(q[2])});x.push(null);y.push(null);z.push(null)});return{x,y,z}},tabs=root.querySelector('#tabs'),eps=root.querySelector('#episodes'),slider=root.querySelector('#slider');let cond='sphere_ood',episode=1,frame=2;function cyl(c){let x=[],y=[],z=[],i=[],n=16,[lo,hi]=D.bounds[2];c.forEach((q,k)=>{let b=x.length;for(let j=0;j<n;j++){let a=2*Math.PI*j/n;x.push(q[0]+q[2]*Math.cos(a),q[0]+q[2]*Math.cos(a));y.push(q[1]+q[2]*Math.sin(a),q[1]+q[2]*Math.sin(a));z.push(lo,hi)}for(let j=0;j<n;j++){let a=b+2*j,bb=b+2*((j+1)%n);i.push(a,bb,a+1,bb,bb+1,a+1)}});return{type:'mesh3d',x,y,z,i:i.filter((_,j)=>j%3===0),j:i.filter((_,j)=>j%3===1),k:i.filter((_,j)=>j%3===2),color:C('--gold'),opacity:.27,hoverinfo:'skip'}}function sph(s){let x=[],y=[],z=[];for(let a=0;a<=14;a++){let p=Math.PI*a/14,xx=[],yy=[],zz=[];for(let b=0;b<=20;b++){let t=2*Math.PI*b/20;xx.push(s[0]+s[3]*Math.sin(p)*Math.cos(t));yy.push(s[1]+s[3]*Math.sin(p)*Math.sin(t));zz.push(s[2]+s[3]*Math.cos(p))}x.push(xx);y.push(yy);z.push(zz)}return{type:'surface',x,y,z,showscale:false,opacity:.34,colorscale:[[0,C('--gold')],[1,C('--gold')]],hoverinfo:'skip'}}function drawControls(){tabs.innerHTML='';Object.keys(D.conditions).forEach(k=>{let b=document.createElement('button');b.textContent=D.conditions[k].label;b.setAttribute('aria-pressed',k===cond);b.onclick=()=>{cond=k;episode=0;frame=0;render()};tabs.append(b)});eps.innerHTML='';D.conditions[cond].cases.forEach((q,i)=>{let b=document.createElement('button');b.textContent=`ep ${i} · ${q.status}`;b.className=q.status.toLowerCase();b.setAttribute('aria-pressed',i===episode);b.onclick=()=>{episode=i;frame=0;render()};eps.append(b)})}function render(){drawControls();let c=D.conditions[cond],q=c.cases[episode],f=q.frames[Math.max(0,Math.min(frame,q.frames.length-1))];frame=Math.max(0,Math.min(frame,q.frames.length-1));slider.max=q.frames.length-1;slider.value=frame;let unsafe=[],safe=[];f.paths.forEach((p,i)=>{if(i&& (f.risk[i].collision||f.risk[i].oob))unsafe.push(p);else if(i)safe.push(p)});let raw=f.paths[0],risk=f.risk[0],executed=xyz(q.path),tr=[...(q.spheres.length?[sph(q.spheres[0])]:[]),...(q.cylinders.length?[cyl(q.cylinders)]:[]),{type:'scatter3d',mode:'lines',...join(unsafe),line:{color:C('--red'),width:2},opacity:.22,hoverinfo:'skip'},{type:'scatter3d',mode:'lines',...join(safe),line:{color:C('--blue'),width:2},opacity:.32,hoverinfo:'skip'},{type:'scatter3d',mode:'lines+markers',...xyz(raw),line:{color:C('--gold'),width:6},marker:{size:2,color:C('--gold')},name:'raw draw'},{type:'scatter3d',mode:'lines',...executed,line:{color:q.status==='SUCCESS'?C('--green'):C('--red'),width:7},name:q.status},{type:'scatter3d',mode:'markers',...xyz([D.start,D.goal,q.path[q.path.length-1]]),marker:{size:6,color:[C('--text'),C('--gold'),q.status==='SUCCESS'?C('--green'):C('--red')]},text:['start','goal',q.status],hoverinfo:'text'}];let b=D.bounds;Plotly.react(root.querySelector('#plot'),tr,{margin:{l:0,r:0,t:3,b:0},paper_bgcolor:'rgba(0,0,0,0)',showlegend:false,scene:{xaxis:{title:'x',range:b[0],gridcolor:C('--border')},yaxis:{title:'y',range:b[1],gridcolor:C('--border')},zaxis:{title:'z',range:b[2],gridcolor:C('--border')},aspectmode:'data',camera:{eye:{x:1.45,y:1.35,z:.9}},bgcolor:'rgba(0,0,0,0)'}},{responsive:true,displaylogo:false});let sum=D.summaries[cond].find(x=>x.gamma===.1);root.querySelector('#subtitle').textContent=`γ=0.1 · ${D.contract.raw_deployment}`;root.querySelector('#audit').textContent=`100-episode audit: SR ${(100*sum.SR).toFixed(0)}% · collision ${(100*sum.CR).toFixed(0)}% · OOB ${(100*sum.OOB).toFixed(0)}%`;root.querySelector('#frame').textContent=`snapshot control step ${f.step}`;root.querySelector('#stats').innerHTML=`<span class=stat><b>${q.status}</b>raw outcome</span><span class=stat><b>${q.steps}</b>control steps</span><span class=stat><b>${(100*q.window_validity).toFixed(0)}%</b>certified windows</span><span class=stat><b>${f.ensemble_safe}/64</b>safe H10 support</span><span class=stat><b>${risk.collision||risk.oob?'unsafe':'safe'}</b>actual K=1 H10</span>`}slider.oninput=()=>{frame=+slider.value;render()};root.querySelector('#prev').onclick=()=>{frame--;render()};root.querySelector('#next').onclick=()=>{frame++;render()};render()})()</script>'''.replace('__DATA__', data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)
    size = args.output.stat().st_size
    if size >= 1_000_000:
        raise RuntimeError(f"fragment is too large ({size} bytes)")
    print(json.dumps({"output": str(args.output), "bytes": size}, indent=2))


if __name__ == "__main__":
    main()
