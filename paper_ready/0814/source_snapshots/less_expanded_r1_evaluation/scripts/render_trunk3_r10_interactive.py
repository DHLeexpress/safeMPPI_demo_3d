"""Render the trunk-3 round-10 diagnosis as a compact interactive fragment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def _compact_path(path):
    return [[round(float(value), 4) for value in point] for point in path]


def _load_trend(stage: Path) -> list[dict]:
    candidates = []
    candidates.extend(
        (stage / "coverage_evaluations/initial/quad_high_sr_trunk3").glob(
            "r*_e80/raw_eval.json"
        )
    )
    candidates.extend(
        (stage / "evaluations/initial/quad_high_sr_trunk3").glob(
            "r*/raw_eval.json"
        )
    )
    by_round = {}
    for path in sorted(candidates):
        payload = json.loads(path.read_text())
        for round_text, summary in payload["summary"].items():
            round_i = int(round_text)
            pooled = summary["pooled"]
            row = {
                "r": round_i,
                "n": int(pooled["episodes"]),
                "sr": round(float(pooled["SR"]), 6),
                "cr": round(float(pooled["CR"]), 6),
                "oob": round(float(pooled["OOB"]), 6),
                "to": round(float(pooled["timeout"]), 6),
                "m": {
                    name: int(pooled["route_counts"].get(name, 0))
                    for name in ("below", "above", "left", "right")
                },
            }
            previous = by_round.get(round_i)
            if previous is None or row["n"] < previous["n"]:
                by_round[round_i] = row
    return [by_round[key] for key in sorted(by_round)]


def _payload(diagnosis: dict, stage: Path, task_config: dict) -> dict:
    raw = []
    for row in diagnosis["raw_trajectories"]:
        raw.append({
            "g": row["gamma"],
            "e": row["episode"],
            "s": row["status"],
            "m": row["mode"],
            "q": row["steps"],
            "b": round(float(row["minimum_margin_m"]), 5),
            "f": row["nearest_face"],
            "n": [round(float(value), 4) for value in row["nearest_point"]],
            "p": _compact_path(row["path"]),
            "o": row.get("oob"),
        })
    training = []
    for row in diagnosis["training_trajectories"]:
        training.append({
            "r": row["round"],
            "g": row["gamma"],
            "m": row["mode"],
            "i": row["trajectory_id"],
            "w": row["windows"],
            "b": round(float(row["minimum_margin_m"]), 5),
            "f": row["nearest_face"],
            "p": _compact_path(row["path"]),
        })
    rounds = []
    for row in diagnosis["training_rounds"]:
        rounds.append({
            "r": row["round"],
            "n": row["trajectory_count"],
            "rows": row["row_count"],
            "near": row["near_wall_trajectory_counts_lt_0.10m"],
            "median": {
                key: round(float(value), 5)
                for key, value in row["mode_min_boundary_margin_median"].items()
            },
        })
    gp = []
    for row in diagnosis["collection_and_gp"]:
        gp.append({
            "r": row["round"],
            "buf": row["gp_buffer_by_gamma"],
            "active": row["gp_active_support"]["row_counts_by_mode_gamma"],
            "fast": row["retry_fast_path_contexts"],
            "retry": row["retry_batches_by_gamma"],
            "attempt": row["attempted_episodes_by_gamma"],
            "all": row["all_terminal_success_modes"],
            "keep": row["committed_modes"],
            "loss": round(float(row["positive_loss"]), 6),
            "step": row["optimizer_step"],
        })
    taskspace = task_config["taskspace"]
    return {
        "bounds": diagnosis["taskspace"]["bounds"],
        "start": taskspace["start"][:3],
        "goal": taskspace["goal"],
        "sphere": task_config["obstacles"]["spheres"][0],
        "raw": raw,
        "train": training,
        "rounds": rounds,
        "gp": gp,
        "trend": _load_trend(stage),
    }


FRAGMENT = r'''
<div id="trunk3-r10-diagnosis">
  <style>
    #trunk3-r10-diagnosis { color: var(--foreground); font: 13px/1.35 ui-sans-serif, system-ui, sans-serif; width: 100%; min-width: 0; overflow: hidden; }
    #trunk3-r10-diagnosis * { box-sizing: border-box; }
    #trunk3-r10-diagnosis [hidden] { display: none !important; }
    #trunk3-r10-diagnosis .title { font-size: 20px; font-weight: 680; letter-spacing: -0.02em; margin: 0 0 4px; }
    #trunk3-r10-diagnosis .sub { color: var(--muted-foreground); margin-bottom: 10px; }
    #trunk3-r10-diagnosis .controls { display: flex; flex-wrap: wrap; gap: 8px 18px; align-items: end; padding: 8px 0 10px; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
    #trunk3-r10-diagnosis .control { display: grid; gap: 3px; min-width: 108px; }
    #trunk3-r10-diagnosis label, #trunk3-r10-diagnosis .label { color: var(--muted-foreground); font-size: 11px; font-weight: 650; letter-spacing: .035em; text-transform: uppercase; }
    #trunk3-r10-diagnosis select, #trunk3-r10-diagnosis input[type="range"] { color: var(--foreground); background: transparent; border: 1px solid var(--border); border-radius: 5px; min-height: 30px; padding: 4px 7px; }
    #trunk3-r10-diagnosis .radio { display: flex; gap: 10px; align-items: center; min-height: 30px; }
    #trunk3-r10-diagnosis .radio label { color: var(--foreground); font-size: 12px; font-weight: 500; letter-spacing: 0; text-transform: none; }
    #trunk3-r10-diagnosis .summary { display: flex; flex-wrap: wrap; gap: 6px 18px; padding: 9px 0 2px; min-height: 40px; }
    #trunk3-r10-diagnosis .summary span { white-space: nowrap; }
    #trunk3-r10-diagnosis .summary strong { font-variant-numeric: tabular-nums; }
    #trunk3-r10-diagnosis .plot3d { width: 100%; height: 555px; }
    #trunk3-r10-diagnosis .mini-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; border-top: 1px solid var(--border); padding-top: 10px; }
    #trunk3-r10-diagnosis .mini { min-width: 0; height: 290px; }
    #trunk3-r10-diagnosis .foot { color: var(--muted-foreground); padding-top: 8px; font-size: 12px; }
    @media (max-width: 620px) {
      #trunk3-r10-diagnosis .title { overflow-wrap: anywhere; }
      #trunk3-r10-diagnosis .controls { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 8px 12px; }
      #trunk3-r10-diagnosis .control { min-width: 0; }
      #trunk3-r10-diagnosis select, #trunk3-r10-diagnosis input[type="range"] { width: 100%; }
      #trunk3-r10-diagnosis .plot3d { height: 420px; }
      #trunk3-r10-diagnosis .mini-grid { grid-template-columns: 1fr; }
      #trunk3-r10-diagnosis .mini { height: 265px; }
    }
  </style>
  <div class="title">Trunk-3 r10: raw exits, replay geometry, and candidate support</div>
  <div class="sub">Fixed seed 91000 · NFE 12 · taskspace x [−2.5, 1.3], y [−1.7, 1.8], z [0.1, 1.7]</div>
  <div class="controls">
    <div class="control"><label for="r10-kind">Trajectories</label><select id="r10-kind"><option value="raw">r10 raw evaluation</option><option value="train">committed training</option></select></div>
    <div class="control"><label for="r10-gamma">Gamma</label><select id="r10-gamma"><option value="all">all</option><option>0.1</option><option>0.3</option><option>0.5</option><option value="1">1.0</option></select></div>
    <div class="control"><label for="r10-mode">Mode</label><select id="r10-mode"><option value="all">all</option><option>below</option><option>above</option><option>left</option><option>right</option></select></div>
    <div class="control" id="r10-status-wrap"><label for="r10-status">Outcome</label><select id="r10-status"><option value="all">all</option><option>SUCCESS</option><option>OOB</option><option>COLLISION</option></select></div>
    <div class="control" id="r10-round-wrap" hidden><label for="r10-round">Training round <span id="r10-round-value">10</span></label><input id="r10-round" type="range" min="1" max="10" value="10"></div>
    <div class="control" id="r10-scope-wrap" hidden><span class="label">Scope</span><div class="radio"><label><input type="radio" name="r10-scope" value="exact" checked> exact</label><label><input type="radio" name="r10-scope" value="cumulative"> cumulative</label></div></div>
  </div>
  <div id="r10-summary" class="summary" aria-live="polite"></div>
  <div id="r10-plot" class="plot3d" role="img" aria-label="Interactive 3D single-sphere trajectories and taskspace boundaries"></div>
  <div class="mini-grid">
    <div id="r10-trend" class="mini" role="img" aria-label="Raw success collision and out-of-bounds rates over expansion rounds"></div>
    <div id="r10-support" class="mini" role="img" aria-label="Pre-quota terminal success support by mode over expansion rounds"></div>
  </div>
  <div class="foot">Taskspace verification checks only the first executed segment; the native exponential cost activates only after a predicted knot is already outside. Dashed line in candidate support is the committed quota floor (12 per mode per round).</div>
  <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
  <script>
  (() => {
    const root = document.getElementById('trunk3-r10-diagnosis');
    const D = __DATA__;
    const css = getComputedStyle(root);
    const color = n => css.getPropertyValue(`--viz-series-${n}`).trim();
    const fg = css.getPropertyValue('--foreground').trim();
    const muted = css.getPropertyValue('--muted-foreground').trim();
    const border = css.getPropertyValue('--border').trim();
    const bg = css.getPropertyValue('--background').trim();
    const pop = css.getPropertyValue('--popover').trim();
    const modeColor = {below: color(1), above: color(2), left: color(3), right: color(4)};
    const statusColor = {SUCCESS: color(3), OOB: color(1), COLLISION: color(2), TIMEOUT: color(6)};
    const ids = name => root.querySelector(`#${name}`);
    const kind = ids('r10-kind'), gamma = ids('r10-gamma'), mode = ids('r10-mode');
    const status = ids('r10-status'), round = ids('r10-round'), roundValue = ids('r10-round-value');
    const summary = ids('r10-summary');
    const fmt = (x, n=3) => Number(x).toFixed(n);

    function boxTrace(bounds) {
      const [x,y,z] = bounds, vertices = [
        [x[0],y[0],z[0]],[x[1],y[0],z[0]],[x[0],y[1],z[0]],[x[1],y[1],z[0]],
        [x[0],y[0],z[1]],[x[1],y[0],z[1]],[x[0],y[1],z[1]],[x[1],y[1],z[1]]
      ], edges = [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]];
      const xs=[],ys=[],zs=[];
      edges.forEach(([a,b]) => { [a,b].forEach(i => {xs.push(vertices[i][0]);ys.push(vertices[i][1]);zs.push(vertices[i][2]);}); xs.push(null);ys.push(null);zs.push(null); });
      return {type:'scatter3d',mode:'lines',x:xs,y:ys,z:zs,line:{color:border,width:3,dash:'dot'},hoverinfo:'skip',showlegend:false};
    }
    function sphereTrace(s) {
      const [cx,cy,cz,r]=s, x=[],y=[],z=[],i=[],j=[],k=[], nu=22,nv=13;
      for(let v=0;v<nv;v++){ const ph=Math.PI*v/(nv-1); for(let u=0;u<nu;u++){const th=2*Math.PI*u/nu;x.push(cx+r*Math.sin(ph)*Math.cos(th));y.push(cy+r*Math.sin(ph)*Math.sin(th));z.push(cz+r*Math.cos(ph));}}
      for(let v=0;v<nv-1;v++) for(let u=0;u<nu;u++){const a=v*nu+u,b=v*nu+(u+1)%nu,c=(v+1)*nu+u,d=(v+1)*nu+(u+1)%nu;i.push(a,b);j.push(c,c);k.push(b,d);}
      return {type:'mesh3d',x,y,z,i,j,k,color:muted,opacity:.26,flatshading:true,hovertemplate:'modeled sphere r=0.2905 m<extra></extra>',name:'sphere',showlegend:true};
    }
    function pathTrace(rows, key, name, c) {
      const x=[],y=[],z=[],text=[];
      rows.forEach(row => {
        const label = row.s ? `${row.s} · ${row.m} · γ=${row.g} · margin ${fmt(row.b)} m` : `r${row.r} · ${row.m} · γ=${row.g} · margin ${fmt(row.b)} m`;
        row.p.forEach(p => {x.push(p[0]);y.push(p[1]);z.push(p[2]);text.push(label);});
        x.push(null);y.push(null);z.push(null);text.push('');
      });
      return {type:'scatter3d',mode:'lines',x,y,z,text,hovertemplate:'%{text}<extra></extra>',line:{color:c,width:key==='OOB'?5:3},opacity:key==='OOB'?.9:.62,name,showlegend:true};
    }
    function selectedRows() {
      if (kind.value === 'raw') return D.raw.filter(row => (gamma.value==='all'||String(row.g)===gamma.value) && (mode.value==='all'||row.m===mode.value) && (status.value==='all'||row.s===status.value));
      const scope = root.querySelector('input[name="r10-scope"]:checked').value;
      const ri = +round.value;
      return D.train.filter(row => (scope==='cumulative'?row.r<=ri:row.r===ri) && (gamma.value==='all'||String(row.g)===gamma.value) && (mode.value==='all'||row.m===mode.value));
    }
    function updateSummary(rows) {
      if (kind.value === 'raw') {
        const count = name => rows.filter(r=>r.s===name).length;
        const succ=count('SUCCESS'), oob=count('OOB'), coll=count('COLLISION');
        const yf=rows.filter(r=>r.o&&r.o.face==='y-min').length;
        summary.innerHTML = `<span><strong>${rows.length}</strong> trajectories</span><span>SR <strong>${rows.length?fmt(succ/rows.length):'—'}</strong></span><span>CR <strong>${rows.length?fmt(coll/rows.length):'—'}</strong></span><span>OOB <strong>${rows.length?fmt(oob/rows.length):'—'}</strong></span><span>y-min exits <strong>${yf}/${oob}</strong></span><span>GP r10 <strong>96 rows/mode×γ</strong></span><span>retry fast-K <strong>${D.gp.at(-1).fast.toLocaleString()}</strong> contexts</span>`;
      } else {
        const margins=rows.map(r=>r.b).sort((a,b)=>a-b), near=margins.filter(x=>x<.1).length, med=margins.length?margins[Math.floor(margins.length/2)]:NaN;
        const counts={below:0,above:0,left:0,right:0}; rows.forEach(r=>counts[r.m]++);
        summary.innerHTML = `<span><strong>${rows.length}</strong> committed trajectories</span><span>near wall &lt;0.10 m <strong>${near}</strong></span><span>median wall margin <strong>${Number.isFinite(med)?fmt(med):'—'} m</strong></span><span>b/a/l/r <strong>${counts.below}/${counts.above}/${counts.left}/${counts.right}</strong></span><span>GP r10 <strong>96 rows/mode×γ</strong></span>`;
      }
    }
    function update3D() {
      const rows=selectedRows(), traces=[boxTrace(D.bounds),sphereTrace(D.sphere)];
      const keyName = kind.value==='raw'?'s':'m';
      const keys = [...new Set(rows.map(r=>r[keyName]))];
      keys.forEach(key => traces.push(pathTrace(rows.filter(r=>r[keyName]===key),key,key,kind.value==='raw'?statusColor[key]:modeColor[key])));
      if(kind.value==='raw'){
        const exits=rows.filter(r=>r.o);
        if(exits.length) traces.push({type:'scatter3d',mode:'markers',x:exits.map(r=>r.o.point[0]),y:exits.map(r=>r.o.point[1]),z:exits.map(r=>r.o.point[2]),text:exits.map(r=>`${r.o.face} · v=[${r.o.velocity.map(v=>fmt(v,2)).join(', ')}]`),hovertemplate:'first OOB: %{text}<extra></extra>',marker:{size:5,color:statusColor.OOB,symbol:'x'},name:'first OOB'});
      } else {
        const near=rows.filter(r=>r.b<.1);
        if(near.length) traces.push({type:'scatter3d',mode:'markers',x:near.map(r=>r.n[0]),y:near.map(r=>r.n[1]),z:near.map(r=>r.n[2]),text:near.map(r=>`${r.m} · ${r.f} · margin ${fmt(r.b)} m`),hovertemplate:'near wall: %{text}<extra></extra>',marker:{size:3,color:color(6)},name:'nearest wall point'});
      }
      traces.push({type:'scatter3d',mode:'markers',x:[D.start[0],D.goal[0]],y:[D.start[1],D.goal[1]],z:[D.start[2],D.goal[2]],text:['start','goal'],hovertemplate:'%{text}<extra></extra>',marker:{size:[5,7],color:[fg,color(5)],symbol:['square','diamond']},name:'start / goal'});
      Plotly.react(ids('r10-plot'),traces,{margin:{l:0,r:0,t:8,b:0},paper_bgcolor:'rgba(0,0,0,0)',font:{color:fg,size:12},legend:{orientation:'h',y:1.02,x:0,bgcolor:'rgba(0,0,0,0)'},scene:{aspectmode:'data',xaxis:{title:'x [m]',range:D.bounds[0],backgroundcolor:bg,gridcolor:border,zerolinecolor:border},yaxis:{title:'y [m]',range:D.bounds[1],backgroundcolor:bg,gridcolor:border,zerolinecolor:border},zaxis:{title:'z [m]',range:D.bounds[2],backgroundcolor:bg,gridcolor:border,zerolinecolor:border},camera:{eye:{x:1.45,y:1.45,z:.9}}}},{responsive:true,displaylogo:false});
      updateSummary(rows);
    }
    function lineTrace(name, ys, c, dash='solid', yaxis='y') { return {type:'scatter',mode:'lines+markers',name,x:D.trend.map(d=>d.r),y:ys,line:{color:c,width:2,dash},marker:{size:5},yaxis,hovertemplate:`r%{x}<br>${name}: %{y:.3f}<extra></extra>`}; }
    function drawMinis(){
      Plotly.newPlot(ids('r10-trend'),[
        lineTrace('SR',D.trend.map(d=>d.sr),color(3)),lineTrace('CR',D.trend.map(d=>d.cr),color(2)),lineTrace('OOB',D.trend.map(d=>d.oob),color(1))
      ],{title:{text:'Raw: collision converts to OOB',x:0,font:{size:14}},margin:{l:48,r:12,t:58,b:42},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:fg,size:11},xaxis:{title:'expansion round',gridcolor:border,dtick:2},yaxis:{title:'rate',range:[0,1],gridcolor:border},legend:{orientation:'h',y:1.12,x:.32,bgcolor:'rgba(0,0,0,0)'}},{responsive:true,displaylogo:false});
      const names=['below','above','left','right'];
      const traces=names.map((name,i)=>({type:'scatter',mode:'lines+markers',name,x:D.gp.map(d=>d.r),y:D.gp.map(d=>d.all[name]||0),line:{color:modeColor[name],width:2},marker:{size:5},hovertemplate:`r%{x}<br>${name}: %{y} terminal successes<extra></extra>`}));
      traces.push({type:'scatter',mode:'lines',name:'quota floor',x:D.gp.map(d=>d.r),y:D.gp.map(()=>12),line:{color:border,width:2,dash:'dash'},hovertemplate:'committed quota: 12<extra></extra>'});
      Plotly.newPlot(ids('r10-support'),traces,{title:{text:'Candidate support before quota pruning',x:0,font:{size:13}},margin:{l:52,r:12,t:58,b:42},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:fg,size:11},xaxis:{title:'collection round',gridcolor:border,dtick:1},yaxis:{title:'terminal successes',type:'log',gridcolor:border},legend:{orientation:'h',y:1.14,x:0,bgcolor:'rgba(0,0,0,0)'}},{responsive:true,displaylogo:false});
    }
    function syncControls(){
      const training=kind.value==='train'; ids('r10-status-wrap').hidden=training;ids('r10-round-wrap').hidden=!training;ids('r10-scope-wrap').hidden=!training;update3D();
    }
    [kind,gamma,mode,status,round].forEach(el=>el.addEventListener('input',()=>{roundValue.textContent=round.value;syncControls();}));
    root.querySelectorAll('input[name="r10-scope"]').forEach(el=>el.addEventListener('change',update3D));
    drawMinis(); syncControls();
  })();
  </script>
</div>
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    diagnosis = json.loads(args.diagnosis.read_text())
    task_config = json.loads(args.task_config.read_text())
    data = json.dumps(
        _payload(diagnosis, args.stage, task_config),
        separators=(",", ":"),
        ensure_ascii=True,
    )
    fragment = FRAGMENT.replace("__DATA__", data)
    if re.search(r"<(?:!doctype|html|head|body)\b", fragment, flags=re.I):
        raise ValueError("visualization must remain an HTML fragment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
