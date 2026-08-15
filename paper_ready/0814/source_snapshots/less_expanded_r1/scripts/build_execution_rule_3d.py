#!/usr/bin/env python3
"""Build an interactive 3D K/B/executed execution-rule comparison fragment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_mppi.config import load_config
from safe_mppi.environment import TaskEnvironment


def _rounded(values, decimals: int = 3):
    return np.round(np.asarray(values), decimals).tolist()


def _even_frames(frames: list[dict], count: int = 6) -> list[dict]:
    if len(frames) <= count:
        return frames
    indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[int(index)] for index in np.unique(indices)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--raw-pre", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--focus-arm")
    parser.add_argument("--case-episode", type=int, action="append", default=[])
    parser.add_argument("--model-label", default="PRE400")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sweep = torch.load(args.sweep, map_location="cpu", weights_only=False)
    raw = torch.load(args.raw_pre, map_location="cpu", weights_only=False)
    config = load_config(args.task_config)
    env = TaskEnvironment(config)
    capture = sweep["contract"]["branch_capture"]
    capture_gamma = float(capture["gamma"])
    capture_episode = int(capture["episode"])
    if args.case_episode and not args.focus_arm:
        parser.error("--case-episode requires --focus-arm")
    if args.focus_arm and args.focus_arm not in sweep["summary"]:
        parser.error(f"unknown --focus-arm {args.focus_arm!r}")

    selected = (
        [
            (f"{args.focus_arm}_ep{episode}", args.focus_arm, int(episode))
            for episode in args.case_episode
        ]
        if args.case_episode else [
            (name, name, capture_episode) for name in sweep["summary"]
        ]
    )

    arms = {}
    for name, source_arm, episode in selected:
        summary = sweep["summary"][source_arm]
        row = next(
            item for item in sweep["rows"]
            if item["arm"] == source_arm
            and float(item["gamma"]) == capture_gamma
            and int(item["episode"]) == episode
        )
        if not row["branch_frames"]:
            raise ValueError(
                f"no captured branch frames for {source_arm} episode {episode}"
            )
        frames = []
        for frame in _even_frames(row["branch_frames"]):
            frames.append({
                "step": int(frame["control_step"]),
                "paths": _rounded(frame["proposed_paths"], 3),
                "eligible": np.asarray(frame["eligible"], np.int32).tolist(),
                "chosen": int(frame["chosen"]),
            })
        arms[name] = {
            "spec": summary["spec"],
            "metrics": summary["pooled"],
            "capture": {
                "gamma": capture_gamma,
                "episode": episode,
                "status": row["status"],
                "mode": row["mode"],
                "steps": int(row["steps"]),
                "time_to_goal_s": row["time_to_goal_s"],
                "case_label": (
                    f"{row['status']} · {row['mode']} · episode {episode}"
                    if args.case_episode else None
                ),
                "path": _rounded(row["path"], 4),
                "frames": frames,
            },
        }

    raw_paths = {}
    for mode in ("left", "right"):
        candidates = [
            row for row in raw["rows"]
            if row["status"] == "SUCCESS"
            and row["mode"] == mode
            and float(row["gamma"]) == capture_gamma
        ]
        if not candidates:
            candidates = [
                row for row in raw["rows"]
                if row["status"] == "SUCCESS" and row["mode"] == mode
            ]
        row = candidates[0]
        dense = np.asarray(row["dense_steps"], np.float32).reshape(-1, 3)
        dense = np.concatenate([
            np.asarray(row["states"], np.float32)[:1, :3], dense,
        ])
        raw_paths[mode] = {
            "gamma": float(row["gamma"]),
            "episode": int(row["episode"]),
            "time_to_goal_s": float(row["time_to_goal_s"]),
            "path": _rounded(dense[::5], 4),
        }

    payload = {
        "model_label": args.model_label,
        "contract": sweep["contract"],
        "scene": {
            "start": _rounded(env.start[:3], 4),
            "goal": _rounded(env.goal, 4),
            "bounds": _rounded(env.bounds, 4),
            "sphere": _rounded(env.spheres[0], 4),
        },
        "arms": arms,
        "raw_pre": raw_paths,
    }
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)

    fragment = r'''<div id="execution-rule-lab" class="viz-shell">
  <style>
    #execution-rule-lab {
      --viz-bg: var(--color-background-primary, light-dark(#fff, #181818));
      --viz-surface: var(--color-background-secondary, light-dark(#f5f7fa, #252525));
      --viz-text: var(--color-text-primary, light-dark(#172033, #fff));
      --viz-muted: var(--color-text-secondary, light-dark(#5b6475, #b6bdc9));
      --viz-border: var(--color-border-secondary, light-dark(#d6dbe5, #4b4b4b));
      --viz-series-1: var(--blue, light-dark(#2774d8, #6aa9ff));
      --viz-series-2: var(--orange, light-dark(#e46b45, #ff9975));
      --viz-series-3: var(--green, light-dark(#2a9d78, #55c99c));
      --viz-series-4: var(--purple, light-dark(#9b62d0, #c18df0));
      --viz-focus: var(--yellow, light-dark(#b57d00, #f0b429));
      --viz-danger: var(--red, light-dark(#c23b3b, #ff7676));
      width: 100%; min-height: 600px; color: var(--viz-text);
      background: transparent; font: 13px/1.4 ui-sans-serif, system-ui, sans-serif;
    }
    #execution-rule-lab * { box-sizing: border-box; }
    #execution-rule-lab .viz-head { display:flex; gap:16px; align-items:flex-end; justify-content:space-between; margin:0 0 10px; }
    #execution-rule-lab h2 { font-size:18px; margin:0 0 3px; }
    #execution-rule-lab p { margin:0; color:var(--viz-muted); }
    #execution-rule-lab .arm-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; margin-bottom:9px; }
    #execution-rule-lab .arm-grid.case-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    #execution-rule-lab .arm-card { border:1px solid var(--viz-border); background:var(--viz-surface); color:var(--viz-text); border-radius:9px; padding:9px; text-align:left; cursor:pointer; min-height:76px; }
    #execution-rule-lab .arm-card[aria-pressed="true"] { border:2px solid var(--viz-series-1); padding:8px; }
    #execution-rule-lab .arm-name { font-weight:500; display:block; margin-bottom:4px; }
    #execution-rule-lab .metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:4px; color:var(--viz-muted); }
    #execution-rule-lab .metrics b { color:var(--viz-text); display:block; font-weight:500; }
    #execution-rule-lab .viz-stage { position:relative; border:1px solid var(--viz-border); border-radius:10px; overflow:hidden; background:var(--viz-bg); }
    #execution-rule-lab #branch-plot { width:100%; height:500px; }
    #execution-rule-lab .controls { display:grid; grid-template-columns:auto 1fr auto auto; gap:9px; align-items:center; padding:8px 10px; border-top:1px solid var(--viz-border); background:var(--viz-surface); }
    #execution-rule-lab input[type="range"] { width:100%; }
    #execution-rule-lab .legend { display:flex; flex-wrap:wrap; gap:10px; padding:7px 10px; color:var(--viz-muted); border-top:1px solid var(--viz-border); }
    #execution-rule-lab .dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:4px; }
    #execution-rule-lab .note { margin-top:7px; font-size:12px; }
    @media (max-width: 620px) {
      #execution-rule-lab .arm-grid,
      #execution-rule-lab .arm-grid.case-grid { grid-template-columns:1fr; }
      #execution-rule-lab #branch-plot { height:410px; }
      #execution-rule-lab .controls { grid-template-columns:auto 1fr auto; }
      #execution-rule-lab .controls .frame-readout { grid-column:1/-1; }
      #execution-rule-lab .viz-head { align-items:flex-start; flex-direction:column; }
    }
  </style>
  <div class="viz-head">
    <div><h2>Execution rule branch laboratory</h2><p id="model-contract"></p></div>
    <p id="case-context"></p>
  </div>
  <div class="arm-grid" id="arm-grid"></div>
  <div class="viz-stage">
    <div id="branch-plot" role="img" aria-label="Interactive 3D candidate branches and executed trajectory"></div>
    <div class="controls">
      <button id="prev-frame" aria-label="Previous captured control step">◀</button>
      <input id="frame-slider" type="range" min="0" value="0" step="1" aria-label="Captured control step">
      <button id="next-frame" aria-label="Next captured control step">▶</button>
      <span class="frame-readout" id="frame-readout"></span>
    </div>
    <div class="legend">
      <span><i class="dot" id="ineligible-dot" style="background:var(--viz-muted)"></i><span id="ineligible-label">K proposals</span></span>
      <span><i class="dot" style="background:var(--viz-series-1)"></i>verifier-eligible</span>
      <span><i class="dot" style="background:var(--viz-focus)"></i>chosen H10 branch</span>
      <span><i class="dot" style="background:var(--viz-series-3)"></i>executed closed loop</span>
      <span><i class="dot" style="background:var(--viz-series-4)"></i>raw PRE left/right</span>
    </div>
  </div>
  <p class="note">Slider frames are six evenly spaced snapshots from the captured rollout. The faint set is verifier-ineligible policy sampling; blue passed the unchanged full verifier; yellow is the symmetric execution-cost choice that fixes the next state.</p>
</div>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script>
(() => {
  const DATA = __DATA__;
  const root = document.getElementById('execution-rule-lab');
  const css = getComputedStyle(root);
  const C = n => css.getPropertyValue(n).trim();
  const names = Object.keys(DATA.arms);
  let armName = names[0], frameIndex = 0;
  const grid = root.querySelector('#arm-grid');
  if(names.length===2) grid.classList.add('case-grid');
  const slider = root.querySelector('#frame-slider');
  const readout = root.querySelector('#frame-readout');
  const xyz = points => ({x:points.map(p=>p[0]), y:points.map(p=>p[1]), z:points.map(p=>p[2])});
  const joined = paths => {
    const x=[],y=[],z=[]; paths.forEach(path => { path.forEach(p=>{x.push(p[0]);y.push(p[1]);z.push(p[2]);}); x.push(null);y.push(null);z.push(null); }); return {x,y,z};
  };
  const label = name => ({exp_balanced:'Exponential · balanced',quad_high_sr:'Quadratic · high SR',quad_fast:'Quadratic · fast'})[name] || name;
  names.forEach(name => {
    const arm=DATA.arms[name], m=arm.metrics, b=document.createElement('button');
    b.className='arm-card'; b.dataset.arm=name; b.setAttribute('aria-pressed',name===armName?'true':'false');
    b.innerHTML=arm.capture.case_label
      ? `<span class="arm-name">${arm.capture.case_label}</span><span class="metrics"><span><b>${arm.capture.status}</b>outcome</span><span><b>${arm.capture.steps}</b>steps</span><span><b>${arm.capture.mode}</b>route</span></span>`
      : `<span class="arm-name">${label(name)}</span><span class="metrics"><span><b>${(100*m.SR).toFixed(1)}%</b>SR</span><span><b>${m.mean_success_time_to_goal_s.toFixed(2)}s</b>TtG</span><span><b>${m.route_coverage}/4</b>coverage</span></span>`;
    b.onclick=()=>{ armName=name; frameIndex=0; [...grid.children].forEach(x=>x.setAttribute('aria-pressed',x.dataset.arm===name?'true':'false')); render(); };
    grid.appendChild(b);
  });
  function sphereTrace(){
    const [cx,cy,cz,r]=DATA.scene.sphere, x=[],y=[],z=[];
    for(let i=0;i<=16;i++){ const ph=Math.PI*i/16, xr=[],yr=[],zr=[]; for(let j=0;j<=24;j++){const th=2*Math.PI*j/24;xr.push(cx+r*Math.sin(ph)*Math.cos(th));yr.push(cy+r*Math.sin(ph)*Math.sin(th));zr.push(cz+r*Math.cos(ph));} x.push(xr);y.push(yr);z.push(zr); }
    return {type:'surface',x,y,z,showscale:false,opacity:.35,colorscale:[[0,C('--viz-series-2')],[1,C('--viz-series-2')]],hoverinfo:'skip',name:'modeled sphere'};
  }
  function render(){
    const arm=DATA.arms[armName], frames=arm.capture.frames;
    frameIndex=Math.max(0,Math.min(frameIndex,frames.length-1)); slider.max=Math.max(0,frames.length-1); slider.value=frameIndex;
    const f=frames[frameIndex], eligible=new Set(f.eligible), all=f.paths.map((p,i)=>({p,i}));
    const proposed=joined(all.filter(v=>!eligible.has(v.i)).map(v=>v.p));
    const verified=joined(all.filter(v=>eligible.has(v.i)&&v.i!==f.chosen).map(v=>v.p));
    const chosen=xyz(f.paths[f.chosen]); const executed=xyz(arm.capture.path);
    const left=xyz(DATA.raw_pre.left.path), right=xyz(DATA.raw_pre.right.path);
    const caseMode=!!arm.capture.case_label, ineligibleColor=caseMode?C('--viz-danger'):C('--viz-muted');
    root.querySelector('#model-contract').textContent=`${DATA.model_label} · pure K=B=64 · no beta / no GP · H10, execute action 0`;
    root.querySelector('#case-context').textContent=`γ=${arm.capture.gamma.toFixed(1)} · episode ${arm.capture.episode}`;
    root.querySelector('#ineligible-dot').style.background=ineligibleColor;
    root.querySelector('#ineligible-label').textContent=caseMode?'verifier-ineligible proposals':'K proposals';
    const traces=[sphereTrace(),
      {type:'scatter3d',mode:'lines',...proposed,line:{color:ineligibleColor,width:1},opacity:caseMode?.24:.16,hoverinfo:'skip',name:'ineligible proposals'},
      {type:'scatter3d',mode:'lines',...verified,line:{color:C('--viz-series-1'),width:2},opacity:.38,hoverinfo:'skip',name:'eligible'},
      {type:'scatter3d',mode:'lines+markers',...chosen,line:{color:C('--viz-focus'),width:6},marker:{size:2,color:C('--viz-focus')},name:'chosen H10'},
      {type:'scatter3d',mode:'lines',...executed,line:{color:C('--viz-series-3'),width:7},name:`executed · ${arm.capture.mode}`},
      {type:'scatter3d',mode:'lines',...left,line:{color:C('--viz-series-4'),width:3,dash:'dot'},opacity:.58,name:'raw PRE left'},
      {type:'scatter3d',mode:'lines',...right,line:{color:C('--viz-series-2'),width:3,dash:'dot'},opacity:.58,name:'raw PRE right'},
      {type:'scatter3d',mode:'markers',...xyz([DATA.scene.start,DATA.scene.goal]),marker:{size:6,color:[C('--viz-text'),C('--viz-focus')]},text:['start','goal'],name:'start / goal'}
    ];
    readout.textContent=`step ${f.step} · eligible ${f.eligible.length}/64 · chosen #${f.chosen}`;
    const b=DATA.scene.bounds;
    Plotly.react(root.querySelector('#branch-plot'),traces,{margin:{l:0,r:0,t:4,b:0},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',showlegend:false,scene:{xaxis:{title:'x',range:[b[0][0],b[0][1]],gridcolor:C('--viz-border')},yaxis:{title:'y',range:[b[1][0],b[1][1]],gridcolor:C('--viz-border')},zaxis:{title:'z',range:[b[2][0],b[2][1]],gridcolor:C('--viz-border')},aspectmode:'data',camera:{eye:{x:1.45,y:1.35,z:.9}},bgcolor:'rgba(0,0,0,0)'}},{responsive:true,displaylogo:false});
  }
  slider.oninput=()=>{frameIndex=+slider.value;render();};
  root.querySelector('#prev-frame').onclick=()=>{frameIndex--;render();};
  root.querySelector('#next-frame').onclick=()=>{frameIndex++;render();};
  render();
})();
</script>'''.replace("__DATA__", data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)
    if args.output.stat().st_size >= 1_000_000:
        raise RuntimeError(
            f"visualization exceeds 1 MB: {args.output.stat().st_size} bytes"
        )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "bytes": args.output.stat().st_size,
        "arms": list(arms),
    }, indent=2))


if __name__ == "__main__":
    main()
