#!/usr/bin/env python3
"""Render fixed-bank axis-cylinder rollouts as an interactive 3-D fragment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


TEMPLATE = r'''
<div id="axis-cylinder-round-lab">
  <style>
    #axis-cylinder-round-lab { color: var(--foreground); width: 100%; min-width: 0; overflow: hidden; }
    #axis-cylinder-round-lab h2 { overflow-wrap: anywhere; }
    #axis-cylinder-round-lab .round-lab-plot { width: 100%; height: 570px; }
    #axis-cylinder-round-lab .round-lab-summary { display: flex; flex-wrap: wrap; gap: 6px 18px; padding: 8px 0 2px; }
    #axis-cylinder-round-lab .round-lab-summary span { white-space: nowrap; }
    @media (max-width: 620px) { #axis-cylinder-round-lab .round-lab-plot { height: 440px; } }
  </style>
  <h2 id="round-lab-title"></h2>
  <div class="viz-controls">
    <label class="form-label" for="round-lab-arm">Execution weight
      <select class="form-select" id="round-lab-arm"></select>
    </label>
    <label class="form-label" for="round-lab-round">Checkpoint
      <select class="form-select" id="round-lab-round"></select>
    </label>
    <label class="form-label" for="round-lab-gamma">Gamma
      <select class="form-select" id="round-lab-gamma"><option value="all">all</option></select>
    </label>
    <label class="form-label" for="round-lab-status">Outcome
      <select class="form-select" id="round-lab-status"><option value="all">all</option><option>SUCCESS</option><option>COLLISION</option><option>OOB</option><option>TIMEOUT</option></select>
    </label>
    <label class="form-check form-switch" for="round-lab-compare">
      <input class="form-check-input" id="round-lab-compare" type="checkbox" checked>
      <span class="form-check-label">Overlay PRE2 raw</span>
    </label>
  </div>
  <div id="round-lab-summary" class="round-lab-summary" aria-live="polite"></div>
  <div id="round-lab-plot" class="round-lab-plot" role="img" aria-label="3D fixed-bank raw trajectories for axis-cylinder execution weights"></div>
  <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
  <script>
  (() => {
    const root=document.getElementById('axis-cylinder-round-lab'), D=__DATA__;
    const css=getComputedStyle(root), token=n=>css.getPropertyValue(`--viz-series-${n}`).trim();
    const fg=css.getPropertyValue('--foreground').trim(), muted=css.getPropertyValue('--muted-foreground').trim(), border=css.getPropertyValue('--border').trim(), bg=css.getPropertyValue('--background').trim();
    const modeColors={below:token(1),above:token(2),left:token(3),right:token(4),none:muted};
    const statusColors={COLLISION:css.getPropertyValue('--destructive').trim(),OOB:token(5),TIMEOUT:muted};
    const arm=root.querySelector('#round-lab-arm'), round=root.querySelector('#round-lab-round'), gamma=root.querySelector('#round-lab-gamma'), status=root.querySelector('#round-lab-status'), compare=root.querySelector('#round-lab-compare');
    root.querySelector('#round-lab-title').textContent=D.title;
    Object.keys(D.arms).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=k;arm.appendChild(o)});
    D.gammas.forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=k;gamma.appendChild(o)});
    arm.value=Object.keys(D.arms)[0];
    function fillRounds(){const prior=round.value,keys=Object.keys(D.arms[arm.value]).sort((a,b)=>+a-+b);round.replaceChildren();keys.forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=k===String(D.compareRound)?'PRE2 raw (r0)':`brake100 round ${k}`;round.appendChild(o)});round.value=keys.includes(prior)?prior:(keys.includes(String(D.defaultRound))?String(D.defaultRound):keys[keys.length-1])}
    fillRounds();
    function boxTrace(){const [xb,yb,zb]=D.bounds,vs=[[xb[0],yb[0],zb[0]],[xb[1],yb[0],zb[0]],[xb[0],yb[1],zb[0]],[xb[1],yb[1],zb[0]],[xb[0],yb[0],zb[1]],[xb[1],yb[0],zb[1]],[xb[0],yb[1],zb[1]],[xb[1],yb[1],zb[1]]],es=[[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]],x=[],y=[],z=[];es.forEach(([a,b])=>{[a,b].forEach(i=>{x.push(vs[i][0]);y.push(vs[i][1]);z.push(vs[i][2])});x.push(null);y.push(null);z.push(null)});return {type:'scatter3d',mode:'lines',x,y,z,line:{color:border,width:3,dash:'dot'},hoverinfo:'skip',showlegend:false}}
    function cylinderTrace(){const a=D.goal.map((v,i)=>v-D.start[i]),n=Math.hypot(...a),axis=a.map(v=>v/n),helper=Math.abs(axis[2])<.9?[0,0,1]:[0,1,0],cross=(u,v)=>[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],norm=u=>{const q=Math.hypot(...u);return u.map(v=>v/q)},u=norm(cross(axis,helper)),v=cross(axis,u),x=[],y=[],z=[],i=[],j=[],k=[],nt=28;for(let end=-1;end<=1;end+=2){const c=D.start.map((s,q)=>s+axis[q]*(end<0?-1:n+1));for(let q=0;q<nt;q++){const t=2*Math.PI*q/nt,p=c.map((s,m)=>s+D.radius*(Math.cos(t)*u[m]+Math.sin(t)*v[m]));x.push(p[0]);y.push(p[1]);z.push(p[2])}}for(let q=0;q<nt;q++){const m=(q+1)%nt;i.push(q,q);j.push(nt+q,nt+m);k.push(m,nt+m)}return {type:'mesh3d',x,y,z,i,j,k,color:token(6),opacity:.055,hoverinfo:'skip',name:'2.2 m cylinder'}}
    function sphereTrace(){const [cx,cy,cz,r]=D.sphere,x=[],y=[],z=[],i=[],j=[],k=[],nu=24,nv=14;for(let b=0;b<nv;b++){const p=Math.PI*b/(nv-1);for(let a=0;a<nu;a++){const t=2*Math.PI*a/nu;x.push(cx+r*Math.sin(p)*Math.cos(t));y.push(cy+r*Math.sin(p)*Math.sin(t));z.push(cz+r*Math.cos(p))}}for(let b=0;b<nv-1;b++)for(let a=0;a<nu;a++){const q=b*nu+a,n=b*nu+(a+1)%nu,s=(b+1)*nu+a,u=(b+1)*nu+(a+1)%nu;i.push(q,n);j.push(s,s);k.push(n,u)}return {type:'mesh3d',x,y,z,i,j,k,color:muted,opacity:.28,hoverinfo:'skip',name:'sphere'}}
    function joined(rows){const x=[],y=[],z=[];rows.forEach(q=>{q.path.forEach(p=>{x.push(p[0]);y.push(p[1]);z.push(p[2])});x.push(null);y.push(null);z.push(null)});return{x,y,z}}
    function draw(){
      const key=arm.value, r=round.value, rows=D.arms[key][r], gf=gamma.value, sf=status.value;
      const keep=q=>(gf==='all'||String(q.gamma)===gf)&&(sf==='all'||q.status===sf), shown=rows.filter(keep);
      const traces=[boxTrace(),cylinderTrace(),sphereTrace()];
      const compareRows=compare.checked&&r!==String(D.compareRound)?D.arms[key][String(D.compareRound)].filter(keep):[];
      if(compareRows.length)traces.push({type:'scatter3d',mode:'lines',...joined(compareRows),line:{color:muted,width:2,dash:'dot'},opacity:.24,hoverinfo:'skip',name:'PRE2 raw'});
      shown.forEach(q=>{const c=q.status==='SUCCESS'?modeColors[q.mode]:statusColors[q.status];traces.push({type:'scatter3d',mode:'lines',x:q.path.map(p=>p[0]),y:q.path.map(p=>p[1]),z:q.path.map(p=>p[2]),line:{color:c,width:q.status==='SUCCESS'?3:2},opacity:q.status==='SUCCESS'?.72:.34,showlegend:false,hovertemplate:`γ ${q.gamma}<br>${q.status} · ${q.mode}<br>episode ${q.episode}<extra></extra>`})});
      traces.push({type:'scatter3d',mode:'lines+markers',x:[D.start[0],D.goal[0]],y:[D.start[1],D.goal[1]],z:[D.start[2],D.goal[2]],line:{color:fg,width:3},marker:{size:[5,7],color:[fg,token(6)],symbol:['square','diamond']},name:'start–goal'});
      Plotly.react(root.querySelector('#round-lab-plot'),traces,{margin:{l:0,r:0,t:8,b:0},paper_bgcolor:'rgba(0,0,0,0)',font:{color:fg},showlegend:false,scene:{aspectmode:'data',xaxis:{title:'x [m]',range:D.bounds[0],backgroundcolor:bg,gridcolor:border},yaxis:{title:'y [m]',range:D.bounds[1],backgroundcolor:bg,gridcolor:border},zaxis:{title:'z [m]',range:D.bounds[2],backgroundcolor:bg,gridcolor:border},camera:{eye:{x:1.45,y:1.45,z:.9}}}},{responsive:true,displaylogo:false});
      const m=D.metrics[key][r], counts=m.route_counts, base=D.metrics[key][String(D.compareRound)];
      root.querySelector('#round-lab-summary').innerHTML=`<span>shown <strong>${shown.length}/${rows.length}</strong></span><span>SR <strong>${m.SR.toFixed(3)}</strong>${compareRows.length?` <span class="text-muted">vs PRE2 ${base.SR.toFixed(3)}</span>`:''}</span><span>CR <strong>${m.CR.toFixed(3)}</strong>${compareRows.length?` <span class="text-muted">vs ${base.CR.toFixed(3)}</span>`:''}</span><span>OOB <strong>${m.OOB.toFixed(3)}</strong>${compareRows.length?` <span class="text-muted">vs ${base.OOB.toFixed(3)}</span>`:''}</span><span>routes b/a/l/r <strong>${counts.below}/${counts.above}/${counts.left}/${counts.right}</strong></span>`;
    }
    arm.addEventListener('change',()=>{fillRounds();draw()});
    [round,gamma,status,compare].forEach(el=>el.addEventListener('change',draw)); draw();
  })();
  </script>
</div>
'''


def _round_rows(path: Path) -> dict[str, list[dict]]:
    paths = (
        [path / "raw_trajectories.pt"]
        if (path / "raw_trajectories.pt").is_file()
        else sorted(path.rglob("raw_trajectories.pt"))
    )
    payload = {}
    for raw_path in paths:
        payload.update(torch.load(raw_path, weights_only=False))
    out: dict[str, list[dict]] = {}
    for round_i, rows in payload.items():
        compact = []
        for row in rows:
            states = np.asarray(row["states"], dtype=float)
            # Keep all 160 fixed-bank rollouts while staying within the app's
            # self-contained visualization budget.
            step = max(1, int(np.ceil(len(states) / 14)))
            path_rows = states[::step, :3]
            if not np.array_equal(path_rows[-1], states[-1, :3]):
                path_rows = np.vstack([path_rows, states[-1, :3]])
            compact.append({
                "gamma": float(row["gamma"]),
                "episode": int(row["episode"]),
                "status": str(row["status"]),
                "mode": str(row["mode"]),
                "path": np.round(path_rows, 2).tolist(),
            })
        out[str(int(round_i))] = compact
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="label=eval_dir")
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Axis-cylinder rollout lab")
    parser.add_argument("--compare-round", type=int, default=0)
    parser.add_argument("--default-round", type=int)
    args = parser.parse_args()
    task = json.loads(args.task_config.read_text())
    data = {"arms": {}, "metrics": {}}
    for spec in args.arm:
        label, raw_dir = spec.split("=", 1)
        raw_dir = Path(raw_dir)
        data["arms"][label] = _round_rows(raw_dir)
        summary = {}
        raw_eval_paths = (
            [raw_dir / "raw_eval.json"]
            if (raw_dir / "raw_eval.json").is_file()
            else sorted(raw_dir.rglob("raw_eval.json"))
        )
        for raw_eval_path in raw_eval_paths:
            summary.update(json.loads(raw_eval_path.read_text())["summary"])
        data["metrics"][label] = {
            str(round_i): values["pooled"] for round_i, values in summary.items()
        }
    data.update({
        "title": args.title,
        "compareRound": args.compare_round,
        "defaultRound": args.default_round,
        "gammas": ["0.1", "0.3", "0.5", "1"],
        "bounds": [
            [float(origin), float(origin + size)]
            for origin, size in zip(
                task["taskspace"]["origin"], task["taskspace"]["size"]
            )
        ],
        "start": task["taskspace"]["start"][:3],
        "goal": task["taskspace"]["goal"],
        "sphere": task["obstacles"]["spheres"][0],
        "radius": 1.1,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(args.output)


if __name__ == "__main__":
    main()
