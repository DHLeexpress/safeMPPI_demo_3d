#!/usr/bin/env python3
"""Build the PRE-vs-unguided-expansion interactive 3D audit fragment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.ball_flow_theta import theta_name, trajectory_crossing_theta
from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment


MODE_COLORS = {
    "below": "#56B4E9",
    "above": "#E69F00",
    "left": "#009E73",
    "right": "#CC79A7",
    "none": "#8A94A6",
}


def _round_points(values: np.ndarray) -> list[list[float]]:
    return np.round(np.asarray(values, dtype=np.float64), 4).tolist()


def _pre_trajectories(path: Path) -> tuple[list[dict], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = payload["rows"]
    trajectories = []
    for row in rows:
        if str(row["status"]).upper() != "SUCCESS":
            continue
        trajectories.append({
            "source": "PRE std1.0",
            "gamma": f"{float(row['gamma']):g}",
            "mode": str(row["mode"]),
            "episode": int(row["episode"]),
            "retry": None,
            "points": _round_points(np.asarray(row["states"])[:, :3]),
        })
    per_gamma = {}
    for gamma in (0.1, 0.3, 0.5, 1.0):
        gamma_rows = [row for row in rows if float(row["gamma"]) == gamma]
        successes = sum(
            str(row["status"]).upper() == "SUCCESS" for row in gamma_rows
        )
        per_gamma[f"{gamma:g}"] = {
            "success": successes,
            "episodes": len(gamma_rows),
            "sr": successes / len(gamma_rows),
        }
    return trajectories, {
        "nfe": int(payload["nfe"]),
        "std": float(payload["flow_base_std"]),
        "seed": int(payload["seed"]),
        "episodes": len(rows),
        "successes": len(trajectories),
        "per_gamma": per_gamma,
    }


def _acquired_trajectories(
    path: Path, task_config: Path,
) -> tuple[list[dict], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    env = TaskEnvironment(load_config(task_config))
    trajectories = []
    for (_, gamma, episode), events in payload["trajectories"].items():
        points = np.concatenate([
            np.asarray(env.start[:3], dtype=np.float32)[None],
            np.stack([
                np.asarray(event["robot_after"], dtype=np.float32)[:3]
                for event in events
            ]),
        ])
        mode = theta_name(trajectory_crossing_theta(env, points))
        trajectories.append({
            "source": "Round-1 observed",
            "gamma": f"{float(gamma):g}",
            "mode": mode,
            "episode": int(episode),
            "retry": int(events[-1]["retry_batch"]),
            "points": _round_points(points),
        })
    return trajectories, metadata


def _html(
    trajectories: list[dict],
    pre: dict,
    acquired: dict,
    task_config: dict,
) -> str:
    authoritative = acquired["authoritative_commit_capable_counts"]
    data_json = json.dumps(trajectories, separators=(",", ":"))
    authoritative_json = json.dumps(authoritative, separators=(",", ":"))
    pre_json = json.dumps(pre, separators=(",", ":"))
    taskspace = task_config["taskspace"]
    task_min = np.asarray(taskspace["origin"], dtype=np.float64)
    task_max = task_min + np.asarray(taskspace["size"], dtype=np.float64)
    sphere = task_config["obstacles"]["spheres"][0]
    geometry_json = json.dumps({
        "min": task_min.tolist(),
        "max": task_max.tolist(),
        "start": taskspace["start"][:3],
        "goal": taskspace["goal"][:3],
        "sphere": {
            "center": sphere[:3],
            "radius": sphere[3],
        },
    }, separators=(",", ":"))
    return f"""
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
  .ue3d-root {{ color:#172033; background:linear-gradient(145deg,#f8fbff 0%,#eef4f8 100%); border:1px solid #d9e3ec; border-radius:18px; padding:18px; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; box-shadow:0 12px 32px rgba(37,55,78,.08); }}
  .ue3d-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:14px; margin-bottom:13px; }}
  .ue3d-kicker {{ color:#47637c; font-size:11px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }}
  .ue3d-title {{ margin:3px 0 5px; font-size:23px; line-height:1.15; letter-spacing:-.025em; }}
  .ue3d-sub {{ color:#5b6d80; font-size:12px; max-width:720px; line-height:1.45; }}
  .ue3d-badge {{ flex:none; color:#9b3a22; background:#fff1ea; border:1px solid #f0b9a8; border-radius:999px; padding:7px 10px; font-size:11px; font-weight:800; }}
  .ue3d-cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin:12px 0; }}
  .ue3d-card {{ background:rgba(255,255,255,.88); border:1px solid #dce6ed; border-radius:12px; padding:10px 11px; min-width:0; }}
  .ue3d-card b {{ display:block; font-size:20px; letter-spacing:-.03em; }}
  .ue3d-card span {{ color:#607286; font-size:10px; line-height:1.3; }}
  .ue3d-flow {{ display:flex; align-items:center; gap:6px; overflow:auto; padding:8px 0 12px; scrollbar-width:thin; }}
  .ue3d-step {{ flex:none; background:#fff; border:1px solid #d5e0e8; border-radius:9px; padding:6px 9px; font-size:10px; font-weight:750; color:#405469; white-space:nowrap; }}
  .ue3d-arrow {{ color:#8aa0b2; font-size:13px; }}
  .ue3d-grid {{ display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:12px; align-items:stretch; }}
  .ue3d-main,.ue3d-side {{ background:rgba(255,255,255,.9); border:1px solid #dce6ed; border-radius:14px; }}
  .ue3d-main {{ overflow:hidden; }}
  .ue3d-controls {{ display:flex; gap:7px; flex-wrap:wrap; padding:10px 10px 0; align-items:center; }}
  .ue3d-controls label {{ color:#607286; font-size:10px; font-weight:800; margin-right:2px; }}
  .ue3d-controls select,.ue3d-controls button {{ appearance:none; background:#f7fafc; color:#26364a; border:1px solid #cfdae3; border-radius:8px; padding:6px 8px; font:700 10px/1.1 inherit; cursor:pointer; }}
  #ue3d-plot {{ height:520px; width:100%; }}
  .ue3d-side {{ padding:12px; }}
  .ue3d-side h3 {{ margin:0 0 3px; font-size:13px; }}
  .ue3d-side p {{ margin:0 0 10px; color:#637588; font-size:10px; line-height:1.4; }}
  .ue3d-gamma {{ margin:0 0 10px; }}
  .ue3d-gamma-top {{ display:flex; justify-content:space-between; align-items:baseline; gap:8px; font-size:11px; font-weight:800; }}
  .ue3d-modes {{ display:grid; grid-template-columns:repeat(4,1fr); gap:3px; margin:5px 0 4px; }}
  .ue3d-mode {{ border-radius:6px; padding:4px 2px; text-align:center; color:#fff; font-size:9px; font-weight:850; }}
  .ue3d-track {{ height:5px; background:#e8edf2; border-radius:99px; overflow:hidden; }}
  .ue3d-fill {{ height:100%; background:linear-gradient(90deg,#4b7798,#54a184); border-radius:99px; }}
  .ue3d-pretable {{ margin-top:14px; padding-top:11px; border-top:1px solid #e1e8ee; }}
  .ue3d-pre-row {{ display:grid; grid-template-columns:42px 1fr 42px; gap:7px; align-items:center; margin:6px 0; font-size:10px; }}
  .ue3d-prebar {{ height:5px; background:#e8edf2; border-radius:99px; overflow:hidden; }}
  .ue3d-prebar i {{ display:block; height:100%; background:#647f99; }}
  .ue3d-note {{ margin-top:11px; border-left:3px solid #e78761; background:#fff6f1; padding:8px 9px; color:#704632; font-size:10px; line-height:1.4; border-radius:4px 8px 8px 4px; }}
  .ue3d-legend {{ display:flex; gap:9px; flex-wrap:wrap; padding:0 10px 10px; color:#52677b; font-size:9px; }}
  .ue3d-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:3px; vertical-align:-1px; }}
  @media(min-width:421px) {{ .ue3d-flow {{ flex-wrap:wrap; }} }}
  @media(max-width:760px) {{ .ue3d-grid {{ grid-template-columns:1fr; }} .ue3d-cards {{ grid-template-columns:repeat(2,1fr); }} #ue3d-plot {{ height:450px; }} .ue3d-side {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }} .ue3d-side > h3,.ue3d-side > p,.ue3d-note {{ grid-column:1/-1; }} .ue3d-pretable {{ margin:0; padding:0; border:0; }} }}
  @media(max-width:420px) {{ .ue3d-root {{ padding:12px; border-radius:14px; }} .ue3d-head {{ display:block; }} .ue3d-badge {{ display:inline-block; margin-top:8px; }} .ue3d-title {{ font-size:19px; }} #ue3d-plot {{ height:410px; }} .ue3d-side {{ display:block; }} .ue3d-controls {{ gap:5px; }} }}
</style>
<div class="ue3d-root">
  <div class="ue3d-head">
    <div><div class="ue3d-kicker">Safe flow expansion · diagnostic mechanism</div><h2 class="ue3d-title">PRE std1.0 → unguided round-1 acquisition</h2><div class="ue3d-sub">Drag to orbit, scroll to zoom, and filter by source, gamma, or route. The sphere and task-space box use the same resolved geometry as the rollout code.</div></div>
    <div class="ue3d-badge">ROUND 1 · NOT COMMITTED</div>
  </div>
  <div class="ue3d-cards">
    <div class="ue3d-card"><b>{pre['episodes']}</b><span>PRE raw rollouts · NFE {pre['nfe']}</span></div>
    <div class="ue3d-card"><b>{pre['successes']} · {100*pre['successes']/pre['episodes']:.1f}%</b><span>PRE terminal successes</span></div>
    <div class="ue3d-card"><b>{acquired['terminal_success_episodes']}</b><span>round-1 terminal successes observed</span></div>
    <div class="ue3d-card"><b>0 / 1</b><span>rounds committed · exact q3 incomplete</span></div>
  </div>
  <div class="ue3d-flow"><div class="ue3d-step">flow std 1.0</div><div class="ue3d-arrow">→</div><div class="ue3d-step">K64 · initial B32</div><div class="ue3d-arrow">→</div><div class="ue3d-step">retry verifies all K64</div><div class="ue3d-arrow">→</div><div class="ue3d-step">full-polytope verifier</div><div class="ue3d-arrow">→</div><div class="ue3d-step">P16 / gamma</div><div class="ue3d-arrow">→</div><div class="ue3d-step">q3 mode keeper</div></div>
  <div class="ue3d-grid">
    <div class="ue3d-main">
      <div class="ue3d-controls"><label>SHOW</label><select id="ue3d-source"><option value="both">PRE + round 1</option><option value="PRE std1.0">PRE only</option><option value="Round-1 observed">round 1 only</option></select><label>GAMMA</label><select id="ue3d-gamma"><option value="all">all</option><option>0.1</option><option>0.3</option><option>0.5</option><option>1</option></select><label>MODE</label><select id="ue3d-mode"><option value="all">all</option><option>below</option><option>above</option><option>left</option><option>right</option></select><button id="ue3d-reset">reset view</button></div>
      <div id="ue3d-plot"></div>
      <div class="ue3d-legend"><span><i class="ue3d-dot" style="background:#56B4E9"></i>below</span><span><i class="ue3d-dot" style="background:#E69F00"></i>above</span><span><i class="ue3d-dot" style="background:#009E73"></i>left</span><span><i class="ue3d-dot" style="background:#CC79A7"></i>right</span><span>thin = PRE · thick = observed round 1</span></div>
    </div>
    <div class="ue3d-side" id="ue3d-side"><h3>Authoritative quota state at cap 32</h3><p>Counts are commit-capable terminal successes in below / above / left / right order. Filled cells stop accumulating.</p><div id="ue3d-quota"></div><div class="ue3d-pretable" id="ue3d-pretable"><h3>PRE raw success rate</h3></div><div class="ue3d-note">54 traces are terminal-success observations extracted from the failed round for diagnosis. Because every gamma did not reach 3/3/3/3, none entered a gradient update.</div></div>
  </div>
</div>
<script>
(() => {{
  const trajectories={data_json};
  const quota={authoritative_json};
  const pre={pre_json};
  const geometry={geometry_json};
  const colors={json.dumps(MODE_COLORS, separators=(',', ':'))};
  const order=['below','above','left','right'];
  const short={{below:'B',above:'A',left:'L',right:'R'}};
  const quotaHost=document.getElementById('ue3d-quota');
  Object.keys(quota).sort((a,b)=>+a-+b).forEach(g=>{{
    const c=quota[g].unguided; const total=order.reduce((s,m)=>s+Math.min(3,c[m]),0);
    quotaHost.insertAdjacentHTML('beforeend',`<div class="ue3d-gamma"><div class="ue3d-gamma-top"><span>γ ${{g}}</span><span>${{total}} / 12</span></div><div class="ue3d-modes">${{order.map(m=>`<div class="ue3d-mode" style="background:${{colors[m]}}">${{short[m]}} ${{c[m]}}/3</div>`).join('')}}</div><div class="ue3d-track"><div class="ue3d-fill" style="width:${{100*total/12}}%"></div></div></div>`);
  }});
  const preHost=document.getElementById('ue3d-pretable');
  Object.entries(pre.per_gamma).forEach(([g,v])=>preHost.insertAdjacentHTML('beforeend',`<div class="ue3d-pre-row"><b>γ ${{g}}</b><div class="ue3d-prebar"><i style="width:${{100*v.sr}}%"></i></div><span>${{v.success}}/${{v.episodes}}</span></div>`));
  function sphereTrace() {{
    const x=[],y=[],z=[],nU=27,nV=14,c=geometry.sphere.center,r=geometry.sphere.radius;
    for(let iv=0;iv<nV;iv++){{const rowX=[],rowY=[],rowZ=[],v=Math.PI*iv/(nV-1);for(let iu=0;iu<nU;iu++){{const u=2*Math.PI*iu/(nU-1);rowX.push(c[0]+r*Math.sin(v)*Math.cos(u));rowY.push(c[1]+r*Math.sin(v)*Math.sin(u));rowZ.push(c[2]+r*Math.cos(v));}}x.push(rowX);y.push(rowY);z.push(rowZ);}}
    return {{type:'surface',x,y,z,showscale:false,opacity:.56,colorscale:[[0,'#ffd3c4'],[1,'#e76643']],hoverinfo:'skip',name:'15-inch sphere + inflation'}};
  }}
  function boxTrace() {{
    const a=geometry.min,b=geometry.max,V=[[a[0],a[1],a[2]],[b[0],a[1],a[2]],[b[0],b[1],a[2]],[a[0],b[1],a[2]],[a[0],a[1],b[2]],[b[0],a[1],b[2]],[b[0],b[1],b[2]],[a[0],b[1],b[2]]],E=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]],x=[],y=[],z=[];
    E.forEach(e=>{{e.forEach(i=>{{x.push(V[i][0]);y.push(V[i][1]);z.push(V[i][2]);}});x.push(null);y.push(null);z.push(null);}});
    return {{type:'scatter3d',mode:'lines',x,y,z,line:{{color:'#91a4b4',width:2,dash:'dot'}},hoverinfo:'skip',name:'task-space'}};
  }}
  function pointTrace() {{return {{type:'scatter3d',mode:'markers+text',x:[geometry.start[0],geometry.goal[0]],y:[geometry.start[1],geometry.goal[1]],z:[geometry.start[2],geometry.goal[2]],text:['START','GOAL'],textposition:'top center',marker:{{size:[5,7],color:['#26364a','#d04b4b'],symbol:['circle','diamond']}},hoverinfo:'text',name:'start / goal'}};}}
  const layout={{margin:{{l:0,r:0,t:4,b:0}},paper_bgcolor:'rgba(0,0,0,0)',showlegend:false,scene:{{bgcolor:'#fbfdff',aspectmode:'manual',aspectratio:{{x:1.15,y:1,z:.65}},xaxis:{{title:'x (m)',range:[geometry.min[0]-.1,geometry.max[0]+.1],gridcolor:'#e3eaf0',zeroline:false}},yaxis:{{title:'y (m)',range:[geometry.min[1]-.1,geometry.max[1]+.1],gridcolor:'#e3eaf0',zeroline:false}},zaxis:{{title:'z (m)',range:[geometry.min[2]-.06,geometry.max[2]+.06],gridcolor:'#e3eaf0',zeroline:false}},camera:{{eye:{{x:1.55,y:1.42,z:1.05}}}}}},hoverlabel:{{font:{{size:11}}}}}};
  const config={{responsive:true,displaylogo:false,modeBarButtonsToRemove:['toImage','lasso3d','select2d']}};
  function draw(reset=false) {{
    const source=document.getElementById('ue3d-source').value,gamma=document.getElementById('ue3d-gamma').value,mode=document.getElementById('ue3d-mode').value;
    const traces=[sphereTrace(),boxTrace(),pointTrace()];
    trajectories.filter(t=>(source==='both'||t.source===source)&&(gamma==='all'||t.gamma===gamma)&&(mode==='all'||t.mode===mode)).forEach(t=>{{const p=t.points;traces.push({{type:'scatter3d',mode:'lines',x:p.map(q=>q[0]),y:p.map(q=>q[1]),z:p.map(q=>q[2]),line:{{color:colors[t.mode]||colors.none,width:t.source==='PRE std1.0'?2:5,dash:t.source==='PRE std1.0'?'dot':'solid'}},opacity:t.source==='PRE std1.0'?.38:.76,name:t.mode,hovertemplate:`${{t.source}}<br>γ=${{t.gamma}} · ${{t.mode}}<br>episode ${{t.episode}}${{t.retry===null?'':` · retry ${{t.retry}}`}}<extra></extra>`}});}});
    const next=reset?JSON.parse(JSON.stringify(layout)):layout;
    Plotly.react('ue3d-plot',traces,next,config);
  }}
  ['ue3d-source','ue3d-gamma','ue3d-mode'].forEach(id=>document.getElementById(id).addEventListener('change',()=>draw(false)));
  document.getElementById('ue3d-reset').addEventListener('click',()=>draw(true));
  draw(true);
}})();
</script>
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-rollouts", type=Path, required=True)
    parser.add_argument("--failed-success-traces", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pre_trajectories, pre = _pre_trajectories(args.pre_rollouts)
    acquired_trajectories, acquired = _acquired_trajectories(
        args.failed_success_traces, args.task_config,
    )
    task_config = json.loads(args.task_config.read_text())
    fragment = _html(
        pre_trajectories + acquired_trajectories,
        pre,
        acquired,
        task_config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment + "\n")
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "pre_success_trajectories": len(pre_trajectories),
        "round1_observed_trajectories": len(acquired_trajectories),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
