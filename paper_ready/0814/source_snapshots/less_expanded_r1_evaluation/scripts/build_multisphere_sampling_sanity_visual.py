#!/usr/bin/env python3
"""Build the inline 3D visual for the two PRE2 round-1 sampling arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rounded(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, float):
        return round(value, 5)
    return value


def _path(values: list[list[float]]) -> list[list[float]]:
    sampled = values[::2]
    if sampled[-1] != values[-1]:
        sampled.append(values[-1])
    return _rounded(sampled)


def _payload(summary: dict[str, Any]) -> dict[str, Any]:
    paired = summary["paired"]
    bowling = summary["bowling"]

    def trajectory(row: dict[str, Any], *, arm: str) -> dict[str, Any]:
        result = {
            "gamma": row["gamma"],
            "episode": row["episode"],
            "path": _path(row["path"]),
            "spheres": _rounded(row["spheres"]),
            "clearance": round(row["minimum_clearance_m"], 4),
            "zmin": round(row["minimum_z_m"], 4),
            "zmax": round(row["maximum_z_m"], 4),
            "time": round(row["time_to_goal_s"], 2),
        }
        if arm == "paired":
            result.update({
                "member": row["paired_scene_member_name"],
                "pair": row["paired_scene_id"],
            })
        else:
            result.update({
                "route": row["route"]["code"],
                "stableRoute": row["route"]["stable_code"],
                "routeMargin": round(
                    row["route"]["minimum_decision_margin_m"], 4,
                ),
            })
        return result

    paired_metrics = paired["metrics"]
    bowling_metrics = bowling["metrics"]
    return {
        "start": _rounded(paired["start"]),
        "goal": _rounded(paired["goal"]),
        "bounds": _rounded(paired["bounds"]),
        "paired": {
            "metrics": {
                "attempts": paired_metrics["attempted_episodes"],
                "successes": paired_metrics["terminal_successes"],
                "nvp": paired_metrics["NVP"],
                "retries": {
                    gamma: max(0, int(batches) - 1)
                    for gamma, batches in paired_metrics[
                        "retry_batches_by_gamma"
                    ].items()
                },
                "optimizerSteps": paired_metrics["optimizer_steps"],
                "loss": round(paired_metrics["loss"], 4),
                "gpBuffer": paired_metrics["gp_buffer"],
                "std": paired_metrics["flow_base_std"],
            },
            "diversity": {
                "raw": round(
                    paired["diversity"]["mean_raw_transverse_rms_m"], 4,
                ),
                "recovered": round(
                    paired["diversity"][
                        "mean_rotation_recovered_rms_m"
                    ],
                    4,
                ),
            },
            "trajectories": [
                trajectory(row, arm="paired")
                for row in paired["trajectories"]
            ],
        },
        "bowling": {
            "metrics": {
                "attempts": bowling_metrics["attempted_episodes"],
                "successes": bowling_metrics["terminal_successes"],
                "nvp": bowling_metrics["NVP"],
                "retries": {
                    gamma: max(0, int(batches) - 1)
                    for gamma, batches in bowling_metrics[
                        "retry_batches_by_gamma"
                    ].items()
                },
                "optimizerSteps": bowling_metrics["optimizer_steps"],
                "loss": round(bowling_metrics["loss"], 4),
                "gpBuffer": bowling_metrics["gp_buffer"],
                "std": bowling_metrics["flow_base_std"],
            },
            "diversity": {
                "routes": bowling["diversity"]["unique_stable_route_codes"],
                "withinZ": round(
                    bowling["diversity"]["mean_within_gamma_z_rms_m"], 4,
                ),
                "betweenZ": round(
                    bowling["diversity"]["mean_between_gamma_z_rms_m"], 4,
                ),
                "betweenLateral": round(
                    bowling["diversity"][
                        "mean_between_gamma_transverse_rms_m"
                    ],
                    4,
                ),
            },
            "trajectories": [
                trajectory(row, arm="bowling")
                for row in bowling["trajectories"]
            ],
        },
    }


def _fragment(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
    return f'''<div id="ms-round1-sanity">
  <h2>PRE2 multi-sphere · round 1 committed trajectories</h2>
  <div class="viz-controls" aria-label="3D trajectory controls">
    <label class="form-label" for="ms-arm">Arm
      <select class="form-select" id="ms-arm">
        <option value="paired">Dense-z · exact axis-180 scene pair</option>
        <option value="bowling">Bowling 1+2+3 · fixed scene</option>
      </select>
    </label>
    <label class="form-label" for="ms-view">View
      <select class="form-select" id="ms-view"></select>
    </label>
    <label class="form-check form-switch">
      <input class="form-check-input" id="ms-box" type="checkbox" checked>
      <span class="form-check-label">Task-space box</span>
    </label>
  </div>
  <div class="viz-grid" id="ms-stats" aria-live="polite"></div>
  <div class="msv-plot" id="ms-plot" role="img" aria-label="Interactive 3D obstacle scenes and committed robot trajectories"></div>
  <div class="viz-row text-small" id="ms-detail" aria-live="polite"></div>
  <div class="table-responsive">
    <table class="table table-sm" aria-label="Visible committed trajectory details">
      <thead><tr><th>γ</th><th>Trajectory</th><th class="text-end">Retry batches</th><th class="text-end">Min. clearance</th><th class="text-end">z range</th><th class="text-end">Time</th></tr></thead>
      <tbody id="ms-rows"></tbody>
    </table>
  </div>
  <p class="sr-only">The dense-z arm contains one original and one independently rolled out axis-rotated scene success for each of four gamma values. The bowling arm contains two successes for each gamma.</p>
</div>
<style>
  #ms-round1-sanity .msv-plot {{ width: 100%; height: 560px; min-height: 460px; }}
  @media (max-width: 480px) {{
    #ms-round1-sanity .msv-plot {{ height: 460px; }}
  }}
</style>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script>
(() => {{
  const root = document.getElementById('ms-round1-sanity');
  const DATA = {encoded};
  const armSelect = root.querySelector('#ms-arm');
  const viewSelect = root.querySelector('#ms-view');
  const boxCheck = root.querySelector('#ms-box');
  const plot = root.querySelector('#ms-plot');
  const stats = root.querySelector('#ms-stats');
  const detail = root.querySelector('#ms-detail');
  const rows = root.querySelector('#ms-rows');
  const gammas = [0.1, 0.3, 0.5, 1.0];

  function themeColor(token) {{
    const probe = document.createElement('span');
    probe.style.color = `var(${{token}})`;
    probe.style.display = 'none';
    root.appendChild(probe);
    const value = getComputedStyle(probe).color;
    probe.remove();
    return value;
  }}

  function palette() {{
    return {{
      foreground: themeColor('--foreground'),
      muted: themeColor('--muted-foreground'),
      border: themeColor('--border'),
      series: [1, 2, 3, 4, 5, 6].map(i => themeColor(`--viz-series-${{i}}`)),
    }};
  }}

  function rotate180ToCanonical(point) {{
    const direction = DATA.goal.map((value, i) => value - DATA.start[i]);
    const norm = Math.sqrt(direction.reduce((total, value) => total + value * value, 0));
    const axis = direction.map(value => value / norm);
    const offset = point.map((value, i) => value - DATA.start[i]);
    const projection = offset.reduce((total, value, i) => total + value * axis[i], 0);
    return offset.map((value, i) => DATA.start[i] + 2 * projection * axis[i] - value);
  }}

  function trajectoryTrace(row, arm, index, colors, options = {{}}) {{
    const xyz = options.path || row.path;
    const gammaIndex = gammas.indexOf(row.gamma);
    const isSecond = arm === 'paired' ? row.member === 'axis_180' : index % 2 === 1;
    const defaultLabel = arm === 'paired'
      ? `γ=${{row.gamma}} · ${{row.member === 'axis_180' ? 'axis 180°' : 'original'}}`
      : `γ=${{row.gamma}} · ${{row.route}} · #${{(index % 2) + 1}}`;
    const label = options.label || defaultLabel;
    return {{
      type: 'scatter3d', mode: 'lines', name: label,
      x: xyz.map(p => p[0]), y: xyz.map(p => p[1]), z: xyz.map(p => p[2]),
      line: {{ color: options.color || colors.series[gammaIndex], width: 7, dash: options.dash || (isSecond ? 'dash' : 'solid') }},
      customdata: xyz.map(() => [row.clearance, row.zmin, row.zmax, row.time]),
      hovertemplate: `${{label}}<br>x=%{{x:.2f}} m<br>y=%{{y:.2f}} m<br>z=%{{z:.2f}} m<br>trajectory min clearance=${{row.clearance.toFixed(3)}} m<extra></extra>`,
      legendgroup: label,
    }};
  }}

  function sphereTrace(sphere, color, name, showLegend, index) {{
    const [cx, cy, cz, radius] = sphere;
    const theta = Array.from({{length: 18}}, (_, i) => 2 * Math.PI * i / 17);
    const phi = Array.from({{length: 10}}, (_, i) => Math.PI * i / 9);
    const x = phi.map(p => theta.map(t => cx + radius * Math.sin(p) * Math.cos(t)));
    const y = phi.map(p => theta.map(t => cy + radius * Math.sin(p) * Math.sin(t)));
    const z = phi.map(p => theta.map(() => cz + radius * Math.cos(p)));
    return {{
      type: 'surface', name, x, y, z, opacity: 0.23, showscale: false,
      colorscale: [[0, color], [1, color]],
      hovertemplate: `${{name}} · sphere ${{index + 1}}<br>center=(${{cx.toFixed(2)}}, ${{cy.toFixed(2)}}, ${{cz.toFixed(2)}}) m<br>effective r=${{radius.toFixed(4)}} m<extra></extra>`,
      showlegend: showLegend,
      legendgroup: name,
    }};
  }}

  function taskBox(colors) {{
    const b = DATA.bounds;
    const corners = [
      [b[0][0], b[1][0], b[2][0]], [b[0][1], b[1][0], b[2][0]],
      [b[0][0], b[1][1], b[2][0]], [b[0][1], b[1][1], b[2][0]],
      [b[0][0], b[1][0], b[2][1]], [b[0][1], b[1][0], b[2][1]],
      [b[0][0], b[1][1], b[2][1]], [b[0][1], b[1][1], b[2][1]],
    ];
    const edges = [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]];
    const x = [], y = [], z = [];
    edges.forEach(([a, c]) => {{
      x.push(corners[a][0], corners[c][0], null);
      y.push(corners[a][1], corners[c][1], null);
      z.push(corners[a][2], corners[c][2], null);
    }});
    return {{
      type: 'scatter3d', mode: 'lines', name: 'task-space boundary',
      x, y, z, line: {{color: colors.border, width: 3}},
      hoverinfo: 'skip', showlegend: false,
    }};
  }}

  function endpoints(colors) {{
    return {{
      type: 'scatter3d', mode: 'markers+text', name: 'start / goal',
      x: [DATA.start[0], DATA.goal[0]], y: [DATA.start[1], DATA.goal[1]], z: [DATA.start[2], DATA.goal[2]],
      text: ['start', 'goal'], textposition: ['top center', 'top center'],
      marker: {{size: 6, color: [colors.series[4], colors.series[5]], symbol: ['circle', 'diamond']}},
      textfont: {{color: colors.foreground}}, hovertemplate: '%{{text}}<br>(%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}}) m<extra></extra>',
      showlegend: false,
    }};
  }}

  function visibleRows(arm, view) {{
    const all = DATA[arm].trajectories;
    return view === 'all' ? all : all.filter(row => String(row.gamma) === view);
  }}

  function stat(label, value, context) {{
    return `<div class="card viz-stat"><span class="text-muted">${{label}}</span><span class="viz-stat-value">${{value}}</span><span class="text-small text-muted">${{context}}</span></div>`;
  }}

  function renderStats(arm) {{
    const d = DATA[arm];
    const m = d.metrics;
    const sr = (100 * m.successes / m.attempts).toFixed(1);
    const retries = gammas.map(g => m.retries[String(g)] ?? m.retries[g]).join(' / ');
    if (arm === 'paired') {{
      stats.innerHTML =
        stat('Exact quota', '8 / 8', 'original 4 + axis-180° 4') +
        stat('Terminal success', `${{m.successes}} / ${{m.attempts}}`, `${{sr}}% · NVP ${{m.nvp}}`) +
        stat('Extra retry batches', retries, `γ=.1/.3/.5/1 · canonical RMS ${{d.diversity.recovered.toFixed(3)}} m`);
    }} else {{
      stats.innerHTML =
        stat('Exact quota', '8 / 8', '2 trajectories × 4 γ') +
        stat('Terminal success', `${{m.successes}} / ${{m.attempts}}`, `${{sr}}% · NVP ${{m.nvp}}`) +
        stat('Observed route codes', `${{d.diversity.routes.length}} / 8`, `${{d.diversity.routes.join(' · ')}} · retries ${{retries}}`);
    }}
  }}

  function renderTable(arm, visible) {{
    const retry = DATA[arm].metrics.retries;
    rows.innerHTML = visible.map((row, index) => {{
      const identity = arm === 'paired'
        ? (row.member === 'axis_180' ? 'axis 180°' : 'original')
        : `${{row.route}} · #${{(DATA[arm].trajectories.filter(x => x.gamma === row.gamma).indexOf(row) + 1)}}`;
      return `<tr><td>${{row.gamma}}</td><td>${{identity}}</td><td class="text-end">${{retry[String(row.gamma)] ?? retry[row.gamma]}}</td><td class="text-end">${{row.clearance.toFixed(3)}} m</td><td class="text-end">${{row.zmin.toFixed(2)}}–${{row.zmax.toFixed(2)}} m</td><td class="text-end">${{row.time.toFixed(1)}} s</td></tr>`;
    }}).join('');
  }}

  function render() {{
    const arm = armSelect.value;
    const view = viewSelect.value;
    const colors = palette();
    const visible = visibleRows(arm, view);
    const traces = [];
    if (boxCheck.checked) traces.push(taskBox(colors));
    if (arm === 'paired' && view !== 'all') {{
      const originals = visible.filter(row => row.member === 'original');
      if (originals.length) originals[0].spheres.forEach((sphere, i) => traces.push(sphereTrace(sphere, colors.muted, 'paired scene', i === 0, i)));
    }} else if (arm === 'bowling') {{
      DATA.bowling.trajectories[0].spheres.forEach((sphere, i) => traces.push(sphereTrace(sphere, colors.muted, 'bowling spheres', i === 0, i)));
    }}
    visible.forEach((row) => {{
      const globalIndex = DATA[arm].trajectories.indexOf(row);
      if (arm === 'paired' && view !== 'all') {{
        const canonical = row.member === 'axis_180'
          ? row.path.map(rotate180ToCanonical)
          : row.path;
        traces.push(trajectoryTrace(row, arm, globalIndex, colors, {{
          path: canonical,
          color: row.member === 'axis_180' ? colors.series[1] : colors.series[0],
          dash: row.member === 'axis_180' ? 'dash' : 'solid',
          label: row.member === 'axis_180' ? `γ=${{row.gamma}} · axis180 → canonical` : `γ=${{row.gamma}} · original`,
        }}));
      }} else {{
        traces.push(trajectoryTrace(row, arm, globalIndex, colors));
      }}
    }});
    traces.push(endpoints(colors));
    const note = arm === 'paired'
      ? (view === 'all' ? 'All 8 raw paths · four distinct scene pairs · obstacles hidden in aggregate view' : `γ=${{view}} · axis-180° member is rotated back only for canonical display`)
      : `Fixed bowling scene · std ${{DATA.bowling.metrics.std.toFixed(1)}} · routes are observed lateral signatures, not coverage quotas`;
    detail.innerHTML = `<span>${{note}}</span><span>Round-1 GP buffer = 0; retry retains K/B sampling, but PRE-relative uncertainty uplift is not yet available.</span>`;
    renderStats(arm);
    renderTable(arm, visible);
    Plotly.react(plot, traces, {{
      margin: {{l: 0, r: 0, t: 8, b: 0}},
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: {{color: colors.foreground, size: 12}},
      legend: {{orientation: 'h', x: 0, y: 1.02, xanchor: 'left', yanchor: 'bottom', font: {{color: colors.foreground}}}},
      scene: {{
        xaxis: {{title: 'x (m)', range: DATA.bounds[0], color: colors.foreground, gridcolor: colors.border, zerolinecolor: colors.border, showbackground: false}},
        yaxis: {{title: 'y (m)', range: DATA.bounds[1], color: colors.foreground, gridcolor: colors.border, zerolinecolor: colors.border, showbackground: false}},
        zaxis: {{title: 'z (m)', range: DATA.bounds[2], color: colors.foreground, gridcolor: colors.border, zerolinecolor: colors.border, showbackground: false}},
        aspectmode: 'manual', aspectratio: {{x: 1.0, y: 0.92, z: 0.56}},
        camera: {{eye: {{x: 1.45, y: 1.35, z: 0.9}}}},
        bgcolor: 'rgba(0,0,0,0)',
      }},
      hoverlabel: {{bgcolor: themeColor('--popover'), font: {{color: themeColor('--popover-foreground')}}}},
      uirevision: arm,
    }}, {{responsive: true, displaylogo: false, scrollZoom: true}});
  }}

  function resetViews() {{
    viewSelect.innerHTML = '<option value="all">All 8 committed paths</option>' + gammas.map(g => `<option value="${{g}}">γ = ${{g}}</option>`).join('');
    viewSelect.value = 'all';
    render();
  }}

  armSelect.addEventListener('change', resetViews);
  viewSelect.addEventListener('change', render);
  boxCheck.addEventListener('change', render);
  resetViews();
  new ResizeObserver(() => Plotly.Plots.resize(plot)).observe(root);
  new MutationObserver(() => render()).observe(document.documentElement, {{attributes: true, attributeFilter: ['class', 'style', 'data-theme']}});
}})();
</script>
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_fragment(_payload(summary)))
    print(args.out.resolve())


if __name__ == "__main__":
    main()
