#!/usr/bin/env python3
"""Render the r10 axis-cylinder calibration as an interactive 3D fragment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TEMPLATE = r'''
<div id="r10-axis-cylinder-calibration">
  <style>
    #r10-axis-cylinder-calibration { color: var(--foreground); width: 100%; min-width: 0; overflow: hidden; }
    #r10-axis-cylinder-calibration .axis-summary { display: flex; flex-wrap: wrap; gap: 6px 18px; padding: 8px 0 2px; }
    #r10-axis-cylinder-calibration .axis-summary span { white-space: nowrap; }
    #r10-axis-cylinder-calibration .axis-plot { width: 100%; height: 560px; }
    #r10-axis-cylinder-calibration .axis-note { color: var(--muted-foreground); padding-top: 6px; }
    @media (max-width: 620px) { #r10-axis-cylinder-calibration .axis-plot { height: 430px; } }
  </style>
  <h2>r10 start–goal cylinder execution calibration</h2>
  <div class="text-muted">Diameter 2.2 m · taskspace guard w500 / 0.15 m · K16 · NFE12</div>
  <div class="viz-controls">
    <label class="form-label" for="axis-probe">Lateral context
      <select class="form-select" id="axis-probe"></select>
    </label>
  </div>
  <div id="axis-summary" class="axis-summary" aria-live="polite"></div>
  <div id="axis-plot" class="axis-plot" role="img" aria-label="3D candidate plans under three start-goal cylinder weights"></div>
  <div class="axis-note text-small">Solid paths are the execution-selected H10 candidate. The translucent cylinder is a scale for the terminal radial attraction, not a verifier boundary.</div>
  <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
  <script>
  (() => {
    const root = document.getElementById('r10-axis-cylinder-calibration');
    const D = __DATA__;
    const css = getComputedStyle(root);
    const color = n => css.getPropertyValue(`--viz-series-${n}`).trim();
    const fg = css.getPropertyValue('--foreground').trim();
    const muted = css.getPropertyValue('--muted-foreground').trim();
    const border = css.getPropertyValue('--border').trim();
    const bg = css.getPropertyValue('--background').trim();
    const arms = ['native','w5','w25','w50'];
    const labels = {native:'taskspace only',w5:'cylinder w5',w25:'cylinder w25',w50:'cylinder w50'};
    const colors = {native:border,w5:color(3),w25:color(1),w50:color(2)};
    const select = root.querySelector('#axis-probe');
    D.probes.forEach((p,i) => {
      const o=document.createElement('option'); o.value=i;
      o.textContent=`γ ${p.gamma} · ${p.mode} ${p.status} · episode ${p.episode}`;
      select.appendChild(o);
    });
    select.value = String(D.probes.findIndex(
      p => p.selections.w5?.changed_from_native
    ));
    function cylinderTrace() {
      const start=D.start, axis=D.axis, length=6.0, radius=D.radius_m;
      let helper=Math.abs(axis[2])<.9?[0,0,1]:[0,1,0];
      const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
      const norm=a=>{const q=Math.hypot(...a);return a.map(v=>v/q)};
      const u=norm(cross(axis,helper)), v=cross(axis,u), x=[],y=[],z=[],I=[],J=[],K=[],nt=28;
      for(let end=-1;end<=1;end+=2){
        const center=start.map((s,k)=>s+axis[k]*(end<0?-1.0:length));
        for(let q=0;q<nt;q++){const t=2*Math.PI*q/nt; const p=center.map((c,k)=>c+radius*(Math.cos(t)*u[k]+Math.sin(t)*v[k]));x.push(p[0]);y.push(p[1]);z.push(p[2]);}
      }
      for(let q=0;q<nt;q++){const n=(q+1)%nt;I.push(q,q);J.push(nt+q,nt+n);K.push(n,nt+n);}
      return {type:'mesh3d',x,y,z,i:I,j:J,k:K,color:color(6),opacity:.08,hoverinfo:'skip',name:'2.2 m cylinder',showlegend:true};
    }
    function boxTrace() {
      const [xb,yb,zb]=D.bounds, vs=[[xb[0],yb[0],zb[0]],[xb[1],yb[0],zb[0]],[xb[0],yb[1],zb[0]],[xb[1],yb[1],zb[0]],[xb[0],yb[0],zb[1]],[xb[1],yb[0],zb[1]],[xb[0],yb[1],zb[1]],[xb[1],yb[1],zb[1]]], es=[[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]],x=[],y=[],z=[];
      es.forEach(([a,b])=>{[a,b].forEach(i=>{x.push(vs[i][0]);y.push(vs[i][1]);z.push(vs[i][2]);});x.push(null);y.push(null);z.push(null);});
      return {type:'scatter3d',mode:'lines',x,y,z,line:{color:border,width:3,dash:'dot'},hoverinfo:'skip',showlegend:false};
    }
    function sphereTrace() {
      const [cx,cy,cz,r]=D.sphere,x=[],y=[],z=[],I=[],J=[],K=[],nu=22,nv=13;
      for(let b=0;b<nv;b++){const ph=Math.PI*b/(nv-1);for(let a=0;a<nu;a++){const th=2*Math.PI*a/nu;x.push(cx+r*Math.sin(ph)*Math.cos(th));y.push(cy+r*Math.sin(ph)*Math.sin(th));z.push(cz+r*Math.cos(ph));}}
      for(let b=0;b<nv-1;b++)for(let a=0;a<nu;a++){const p=b*nu+a,q=b*nu+(a+1)%nu,s=(b+1)*nu+a,t=(b+1)*nu+(a+1)%nu;I.push(p,q);J.push(s,s);K.push(q,t);}
      return {type:'mesh3d',x,y,z,i:I,j:J,k:K,color:muted,opacity:.28,hovertemplate:'modeled sphere r=0.2905 m<extra></extra>',name:'sphere'};
    }
    function draw() {
      const p=D.probes[+select.value], traces=[boxTrace(),cylinderTrace(),sphereTrace()];
      arms.forEach(arm=>{const s=p.selections[arm];if(!s)return;const path=s.planned_path;traces.push({type:'scatter3d',mode:'lines+markers',x:path.map(q=>q[0]),y:path.map(q=>q[1]),z:path.map(q=>q[2]),line:{color:colors[arm],width:arm==='native'?4:6,dash:arm==='native'?'dot':'solid'},marker:{color:colors[arm],size:2},name:labels[arm],hovertemplate:`${labels[arm]}<br>axis max ${s.predicted_max_axis_distance_m.toFixed(3)} m<br>goal ${s.terminal_goal_distance_m.toFixed(3)} m<extra></extra>`});});
      traces.push({type:'scatter3d',mode:'lines+markers',x:[D.start[0],D.goal[0]],y:[D.start[1],D.goal[1]],z:[D.start[2],D.goal[2]],line:{color:fg,width:3},marker:{size:[5,7],color:[fg,color(5)],symbol:['square','diamond']},name:'start–goal axis'});
      Plotly.react(root.querySelector('#axis-plot'),traces,{margin:{l:0,r:0,t:8,b:0},paper_bgcolor:'rgba(0,0,0,0)',font:{color:fg},legend:{orientation:'h',y:1.02,x:0},scene:{aspectmode:'data',xaxis:{title:'x [m]',range:D.bounds[0],backgroundcolor:bg,gridcolor:border},yaxis:{title:'y [m]',range:D.bounds[1],backgroundcolor:bg,gridcolor:border},zaxis:{title:'z [m]',range:D.bounds[2],backgroundcolor:bg,gridcolor:border},camera:{eye:{x:1.45,y:1.45,z:.9}}}},{responsive:true,displaylogo:false});
      const changed=arm=>p.selections[arm]?.changed_from_native?'changed':'same';
      root.querySelector('#axis-summary').innerHTML=`<span>source radius <strong>${p.source_max_axis_distance_m.toFixed(3)} m</strong></span><span>eligible <strong>${p.eligible}/16</strong></span><span>w5 <strong>${changed('w5')}</strong></span><span>w25 <strong>${changed('w25')}</strong></span><span>w50 <strong>${changed('w50')}</strong></span><span>global change rates <strong>6.25% / 25% / 31.25%</strong></span>`;
    }
    select.addEventListener('change',draw); draw();
  })();
  </script>
</div>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    task = json.loads(args.task_config.read_text())
    payload = {
        "bounds": calibration["taskspace_bounds"],
        "start": calibration["start"],
        "goal": calibration["goal"],
        "axis": calibration["axis"],
        "radius_m": calibration["radius_m"],
        "sphere": task["obstacles"]["spheres"][0],
        "probes": calibration["probes"],
    }
    fragment = TEMPLATE.replace(
        "__DATA__", json.dumps(payload, separators=(",", ":")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)
    print(args.output)


if __name__ == "__main__":
    main()
